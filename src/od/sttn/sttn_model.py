import math
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseODModel
import numpy as np


def calculate_random_walk_matrix(adj_mx):
    """Returns the random walk adjacency matrix (for D_GCN)."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


class Norm_S(nn.Module):
    """Gaussian head over the destination axis.

    From the spatial features ``(B, hidden, N_origin, N_dest)`` it emits, for
    every (batch, horizon, origin):

      * a mean vector ``loc`` over the ``N`` destinations, and
      * a low-rank-plus-diagonal covariance ``Sigma = L Lᵀ + diag``
        (``L`` is ``N×rank``), which is symmetric positive-definite by
        construction — no eigen-decomposition needed, and stable to train.

    Output: ``loc (B, H, N, N)``, ``Sigma (B, H, N, N, N)`` where ``H`` is the
    forecast horizon (= ``c_out``) and the trailing ``N×N`` is the per-origin
    destination covariance.
    """

    def __init__(self, c_in, c_out, feature, min_vec, rank=4):
        super(Norm_S, self).__init__()
        self.c_out = c_out          # horizon
        self.feature = feature      # N destinations
        self.min_vec = min_vec
        self.rank = rank

        self.loc_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        # low-rank factor: c_out * rank channels per (origin, destination)
        self.fac_conv = nn.Conv2d(c_in, c_out * rank, kernel_size=(1, 1), bias=True)
        # log-diagonal: c_out channels per (origin, destination)
        self.diag_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):  # x (B, hidden, N_origin, N_dest)
        B = x.shape[0]
        N = self.feature
        H = self.c_out

        loc = F.softplus(self.loc_conv(x))           # (B, H, N_o, N_d)

        fac = self.fac_conv(x)                        # (B, H*rank, N_o, N_d)
        fac = fac.view(B, H, self.rank, N, N)         # (B, H, r, N_o, N_d)
        # per (B, H, origin): factor L is (N_dest, rank)
        L = fac.permute(0, 1, 3, 4, 2)                # (B, H, N_o, N_d, r)

        diag = F.softplus(self.diag_conv(x)) + self.min_vec  # (B, H, N_o, N_d)
        diag = diag.permute(0, 1, 2, 3)                       # (B, H, N_o, N_d)

        # Sigma = L Lᵀ + diag·I  -> (B, H, N_o, N_d, N_d)
        sigma = torch.matmul(L, L.transpose(-1, -2))
        eye = torch.eye(N, device=x.device).view(1, 1, 1, N, N)
        sigma = sigma + diag.unsqueeze(-1) * eye

        return loc, sigma


class MDGCN(nn.Module):
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
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        batch_size, input_size, num_node, feature = X.shape
        supports = [A_q, A_h]

        x0 = X.permute(3, 2, 1, 0)
        x0 = torch.reshape(x0, shape=[feature, num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        for support in supports:
            s = support.unsqueeze(0).expand(x0.shape[0], -1, -1)
            x1 = torch.matmul(s, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.matmul(s, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, feature, num_node, input_size, batch_size]
        )
        x = x.permute(1, 4, 2, 3, 0)
        x = torch.reshape(
            x, shape=[feature, batch_size, num_node, input_size * self.num_matrices]
        )
        x = torch.matmul(x, self.Theta1)
        x += self.bias
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "selu":
            x = F.selu(x)

        x = x.permute(1, 3, 2, 0)
        return x


class STTN(BaseODModel):
    """Spatial-Temporal Transformer-style network with a multivariate-Gaussian
    output for OD demand: it predicts a mean OD matrix ``loc`` and, for every
    origin, a full ``N×N`` covariance over the destinations (positive-definite
    by eigenvalue clamping).  Trained with the multivariate-normal NLL (``MGAU``
    loss).  Channel-as-batch over the 3 mobility channels (see
    :class:`base.model.BaseODModel`); driven by :class:`STTN_Engine`.
    """

    cqr_compatible = False

    def __init__(
        self,
        A,
        node_num,
        hidden_dim_t,
        hidden_dim_s,
        rank_t,
        rank_s,
        num_timesteps_input,
        num_timesteps_output,
        device,
        input_dim,
        output_dim,
        seq_len,
        horizon,
        min_vec,
        **args
    ):
        super(STTN, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)

        self.num_feature = node_num

        self.SC1 = MDGCN(num_timesteps_input, hidden_dim_s, 3)
        self.SC2 = MDGCN(hidden_dim_s, rank_s, 2, activation="linear")
        self.SC3 = MDGCN(rank_s, hidden_dim_s, 2)
        self.SGau = Norm_S(hidden_dim_s, num_timesteps_output, self.num_feature, min_vec)

        self.A = A
        A_q = torch.from_numpy(calculate_random_walk_matrix(self.A).T.astype("float32"))
        A_h = torch.from_numpy(calculate_random_walk_matrix(self.A.T).T.astype("float32"))
        self.register_buffer("A_q", A_q)
        self.register_buffer("A_h", A_h)

    def forward(self, X, label=None):
        """X (B, T, N, N, D) -> (loc, Sigma).

        ``loc``  : (B, horizon, N, N, D)         — mean OD matrix per channel
        ``Sigma``: (B, horizon, N, N, N, D)      — per-origin destination cov.
        """
        X, b, d, squeeze_back = self._fold_channels(X)

        X_s1 = self.SC1(X, self.A_q, self.A_h)
        X_s2 = self.SC2(X_s1, self.A_q, self.A_h)
        X_s3 = self.SC3(X_s2, self.A_q, self.A_h)
        loc, sigma = self.SGau(X_s3)  # loc (B', H, N, N), sigma (B', H, N, N, N)

        loc = self._unfold_channels(loc, b, d, squeeze_back)
        sigma = self._unfold_sigma(sigma, b, d, squeeze_back)
        return loc, sigma

    @staticmethod
    def _unfold_sigma(sigma, b, d, squeeze_back):
        """Unfold a covariance tensor ``(B*D, H, N, N, N)`` back to
        ``(B, H, N, N, N, D)`` (or drop the channel axis when single-channel)."""
        _, h, n, m, m2 = sigma.shape
        sigma = sigma.reshape(b, d, h, n, m, m2).permute(0, 2, 3, 4, 5, 1)
        if squeeze_back:
            sigma = sigma.squeeze(-1)
        return sigma
