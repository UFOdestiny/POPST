"""UQGNN — Uncertainty Quantification Graph Neural Network.

Faithful re-implementation of the architecture in
"Uncertainty-aware Graph Neural Networks for Multivariate Spatiotemporal
Prediction" (arXiv:2508.08551v2).

The model predicts, for every region (node) and forecast step, a *multivariate
Gaussian* over the ``M`` urban phenomena (e.g. taxi / TNP / bike demand):

    p(y_{n} | X) = N( mu_{n} , Sigma_{n} ),   mu in R^M ,  Sigma in R^{MxM} (PD)

It has two parts:

* **ISTE** — Interaction-aware SpatioTemporal Embedding.  A spatial branch
  (Multivariate Diffusion GCN, ``MDGCN``) and a temporal branch
  (Interaction-aware Temporal Conv, ``ITCN``) each produce an embedding of
  shape ``(B, N, M, e)``.  They are fused with a **Hadamard product**
  ``E = E_s ⊙ E_t``.

* **MPP** — Multivariate Probabilistic Prediction.  Two heads map the fused
  embedding to the mean ``mu`` and to a vector that is assembled into a
  positive-definite covariance ``Sigma`` (Algorithm 1: symmetric fill →
  eigen-decomposition → eigenvalue clamping → reconstruction).

``forward`` returns ``(mu, Sigma)``; the engine treats the second element as
the per-prediction covariance and trains with the multivariate-Gaussian NLL
(``MGAU`` loss in :mod:`base.metrics`).
"""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def calculate_random_walk_matrix(adj_mx):
    """Row-normalised (random-walk) transition matrix, returned dense."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


# ---------------------------------------------------------------------------
# Spatial branch: Multivariate Diffusion GCN
# ---------------------------------------------------------------------------

class MDGCN(nn.Module):
    """Multivariate diffusion graph convolution.

    Diffuses information over the bidirectional random-walk supports while
    keeping the node (``N``) and phenomenon (``M``) axes intact, mapping the
    ``in_channels`` axis to ``out_channels``.

    Input/Output: ``(B, in_channels, N, M)`` -> ``(B, out_channels, N, M)``.
    """

    def __init__(self, in_channels, out_channels, orders, activation="relu"):
        super(MDGCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(in_channels * self.num_matrices, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        return torch.cat([x, x_.unsqueeze(0)], dim=0)

    def forward(self, X, A_q, A_h):
        batch_size, input_size, num_node, feature = X.shape
        supports = [A_q, A_h]

        x0 = X.permute(3, 2, 1, 0)  # (M, N, in_channels, B)
        x0 = torch.reshape(x0, shape=[feature, num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        # Chebyshev-style diffusion expansion over each support
        for support in supports:
            s = support.unsqueeze(0).expand(x0.shape[0], -1, -1)
            x1 = torch.matmul(s, x0)
            x = self._concat(x, x1)
            for _ in range(2, self.orders + 1):
                x2 = 2 * torch.matmul(s, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, feature, num_node, input_size, batch_size]
        )
        x = x.permute(1, 4, 2, 3, 0)  # (M, B, N, in_channels, num_matrices)
        x = torch.reshape(
            x, shape=[feature, batch_size, num_node, input_size * self.num_matrices]
        )
        x = torch.matmul(x, self.Theta1) + self.bias  # (M, B, N, out_channels)

        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "selu":
            x = F.selu(x)

        return x.permute(1, 3, 2, 0)  # (B, out_channels, N, M)


# ---------------------------------------------------------------------------
# Temporal branch: Interaction-aware Temporal Convolution
# ---------------------------------------------------------------------------

class ITCN(nn.Module):
    """Interaction-aware temporal convolution.

    Gated dilated 1-D convolutions over the time axis, applied identically to
    every node, that *mix the phenomena* (interaction-aware) while producing a
    per-phenomenon embedding.

    Input ``(B, T, N, M)`` -> embedding ``(B, N, M, e)``.
    """

    def __init__(self, num_feature, emb_dim, kernel_size=3, layers=2):
        super(ITCN, self).__init__()
        self.M = num_feature
        self.e = emb_dim
        out_ch = num_feature * emb_dim

        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        in_ch, dilation = num_feature, 1
        for _ in range(layers):
            pad = dilation * (kernel_size - 1) // 2  # keep the time length fixed
            self.filters.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
            )
            self.gates.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
            )
            in_ch, dilation = out_ch, dilation * 2

    def forward(self, X):
        B, T, N, M = X.shape
        # (B, T, N, M) -> (B*N, M, T): each node is a sequence, M are channels
        h = X.permute(0, 2, 3, 1).reshape(B * N, M, T)
        for filt, gate in zip(self.filters, self.gates):
            h = torch.tanh(filt(h)) * torch.sigmoid(gate(h))
        h = h.mean(dim=-1)  # aggregate over time -> (B*N, M*e)
        return h.reshape(B, N, self.M, self.e)


# ---------------------------------------------------------------------------
# Covariance assembly (Algorithm 1)
# ---------------------------------------------------------------------------

def build_covariance(vec, feature, index, min_vec):
    """Assemble a positive-definite covariance from its lower-triangular vector.

    ``vec`` holds the ``M(M+1)/2`` free entries; they are written into both
    triangles of a symmetric matrix, which is then projected onto the PD cone
    by clamping its eigenvalues at ``min_vec``.

    ``vec`` shape ``(..., M(M+1)/2)`` -> covariance ``(..., M, M)``.
    """
    z = torch.zeros(*vec.shape[:-1], feature, feature,
                    device=vec.device, dtype=vec.dtype)
    z[..., index[0], index[1]] = vec
    z[..., index[1], index[0]] = vec

    eigval, eigvec = torch.linalg.eigh(z)
    eigval = torch.clamp(eigval, min=min_vec)
    cov = torch.matmul(eigvec, torch.diag_embed(eigval))
    cov = torch.matmul(cov, eigvec.transpose(-2, -1))
    return 0.5 * (cov + cov.transpose(-2, -1))  # enforce exact symmetry


# ---------------------------------------------------------------------------
# UQGNN
# ---------------------------------------------------------------------------

class UQGNN(BaseModel):
    # UQGNN emits its own multivariate-Gaussian distribution (mu, Sigma) and so
    # is incompatible with the CQR engine (which expects point/quantile output).
    cqr_compatible = False

    def __init__(self, A, node_num, hidden_dim_s, hidden_dim_t, emb_dim,
                 kernel_size, temporal_layers, num_timesteps_output, device,
                 input_dim, output_dim, seq_len, min_vec, **args):
        super(UQGNN, self).__init__(node_num, input_dim, output_dim, seq_len, **args)

        self.num_feature = input_dim          # M phenomena
        self.emb_dim = emb_dim                # e
        self.min_vec = min_vec
        H = num_timesteps_output              # forecast horizon
        M = input_dim
        self.half = M * (M + 1) // 2          # free covariance entries

        # ---- ISTE: spatial branch (MDGCN) ----
        self.SC1 = MDGCN(seq_len, hidden_dim_s, orders=3)
        self.SC2 = MDGCN(hidden_dim_s, emb_dim, orders=2, activation="relu")

        # ---- ISTE: temporal branch (ITCN) ----
        self.TC = ITCN(M, emb_dim, kernel_size=kernel_size, layers=temporal_layers)

        # ---- MPP: mean + covariance heads ----
        self.mean_head = nn.Linear(emb_dim, H)
        self.cov_head = nn.Linear(M * emb_dim, H * self.half)

        # Random-walk supports (forward / backward), kept as buffers
        A = np.asarray(A, dtype=np.float32)
        A_q = calculate_random_walk_matrix(A).T.astype("float32")
        A_h = calculate_random_walk_matrix(A.T).T.astype("float32")
        self.register_buffer("A_q", torch.from_numpy(A_q))
        self.register_buffer("A_h", torch.from_numpy(A_h))
        self.register_buffer("tri_idx", torch.triu_indices(M, M))

        self.to(device=device)

    def forward(self, X, label=None):
        # X: (B, T, N, M)
        B, T, N, M = X.shape

        # Spatial embedding E_s: (B, N, M, e)
        s = self.SC1(X, self.A_q, self.A_h)
        s = self.SC2(s, self.A_q, self.A_h)
        E_s = s.permute(0, 2, 3, 1)

        # Temporal embedding E_t: (B, N, M, e)
        E_t = self.TC(X)

        # Hadamard fusion
        E = E_s * E_t  # (B, N, M, e)

        # Mean head: e -> H, positive (counts are non-negative)
        mu = self.mean_head(E)                 # (B, N, M, H)
        mu = F.softplus(mu).permute(0, 3, 1, 2)  # (B, H, N, M)

        # Covariance head: per-node (M*e) -> H * M(M+1)/2
        z = self.cov_head(E.reshape(B, N, M * self.emb_dim))  # (B, N, H*half)
        z = z.reshape(B, N, mu.shape[1], self.half).permute(0, 2, 1, 3)  # (B, H, N, half)
        sigma = build_covariance(z, M, self.tri_idx, self.min_vec)  # (B, H, N, M, M)

        return mu, sigma
