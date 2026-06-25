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


class NBNorm_ZeroInflated_T(nn.Module):
    """Temporal head emitting (n, p, pi) of a zero-inflated NB, one set per
    (horizon, node, destination) cell.  ``n = softplus``, ``p = sigmoid``,
    ``pi = sigmoid`` (Zhuang et al., KDD'21)."""

    def __init__(self, c_in, c_out, seq_len):
        super().__init__()
        self.n_conv = nn.Conv2d(c_in, c_out, kernel_size=(seq_len, 1), bias=True)
        self.p_conv = nn.Conv2d(c_in, c_out, kernel_size=(seq_len, 1), bias=True)
        self.pi_conv = nn.Conv2d(c_in, c_out, kernel_size=(seq_len, 1), bias=True)
        self.out_dim = c_out

    def forward(self, x):  # x (B, T, N, F)
        x = x.permute(0, 2, 1, 3)  # (B, F, T, N) -> conv collapses the time axis
        n = F.softplus(self.n_conv(x))
        p = torch.sigmoid(self.p_conv(x))
        pi = torch.sigmoid(self.pi_conv(x))
        # (B, c_out, 1, N) -> (B, 1(=horizon), N, c_out)
        n = n.permute(0, 2, 1, 3)
        p = p.permute(0, 2, 1, 3)
        pi = pi.permute(0, 2, 1, 3)
        return n, p, pi


class NBNorm_ZeroInflated_S(nn.Module):
    """Spatial head emitting (n, p, pi) of a zero-inflated NB."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.n_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        self.p_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        self.pi_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        self.out_dim = c_out

    def forward(self, x):  # x (B, F, N, horizon)
        n = F.softplus(self.n_conv(x))
        p = torch.sigmoid(self.p_conv(x))
        pi = torch.sigmoid(self.pi_conv(x))
        return n, p, pi


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


class ITCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation="relu"):
        super(ITCN, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        batch_size, seq_len, num_nodes, num_features = X.shape

        Xf = X
        inv_idx = torch.arange(Xf.size(1) - 1, -1, -1).long().to(device=Xf.device)
        Xb = Xf.index_select(1, inv_idx)

        Xf = Xf.permute(0, 2, 3, 1)
        Xb = Xb.permute(0, 2, 3, 1)

        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape(
            [batch_size, seq_len - self.kernel_size + 1, self.out_channels, num_features]
        )

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape(
            [batch_size, seq_len - self.kernel_size + 1, self.out_channels, num_features]
        )

        rec = torch.zeros(
            [batch_size, self.kernel_size - 1, self.out_channels, num_features]
        ).to(device=Xf.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)

        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=Xf.device)
        outb = outb.index_select(1, inv_idx)
        if self.activation == "relu":
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == "sigmoid":
            out = F.sigmoid(outf) + F.sigmoid(outb)
        else:
            out = outf + outb
        return out


class STZINB(BaseODModel):
    """Spatial-Temporal Zero-Inflated Negative Binomial network (Zhuang et al.,
    KDD 2021) for sparse OD demand.

    Twin branches — a temporal inception TCN and a spatial diffusion GCN — each
    emit zero-inflated NB parameters ``(n, p, pi)``; the two estimates are fused
    multiplicatively.  Trained by the ZINB negative log-likelihood (see
    :func:`base.metrics.zinb_nll`); the point prediction is the ZINB mean
    ``E[y] = (1-pi)·n·(1-p)/p``.  Channel-as-batch over the 3 mobility channels
    (see :class:`base.model.BaseODModel`); driven by :class:`STZINB_Engine`.
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
        **args
    ):
        super(STZINB, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)

        self.TC1 = ITCN(node_num, hidden_dim_t, kernel_size=3)
        self.TC2 = ITCN(hidden_dim_t, rank_t, kernel_size=3, activation="linear")
        self.TC3 = ITCN(rank_t, hidden_dim_t, kernel_size=3)
        self.TNB = NBNorm_ZeroInflated_T(hidden_dim_t, node_num, self.seq_len)

        self.SC1 = MDGCN(num_timesteps_input, hidden_dim_s, 3)
        self.SC2 = MDGCN(hidden_dim_s, rank_s, 2, activation="linear")
        self.SC3 = MDGCN(rank_s, hidden_dim_s, 2)
        self.SNB = NBNorm_ZeroInflated_S(hidden_dim_s, num_timesteps_output)

        self.A = A
        A_q = torch.from_numpy(calculate_random_walk_matrix(self.A).T.astype("float32"))
        A_h = torch.from_numpy(calculate_random_walk_matrix(self.A.T).T.astype("float32"))
        self.register_buffer("A_q", A_q)
        self.register_buffer("A_h", A_h)

    def forward(self, X, label=None):
        """X (B, T, N, N, D) -> (n, p, pi), each (B, horizon, N, N, D)."""
        X, b, d, squeeze_back = self._fold_channels(X)

        # temporal branch -> params over node_num destinations
        X_t = self.TC1(X)
        X_t = self.TC2(X_t)
        X_t = self.TC3(X_t)
        n_t, p_t, pi_t = self.TNB(X_t)

        # spatial branch
        X_s = self.SC1(X, self.A_q, self.A_h)
        X_s = self.SC2(X_s, self.A_q, self.A_h)
        X_s = self.SC3(X_s, self.A_q, self.A_h)
        n_s, p_s, pi_s = self.SNB(X_s)

        # fuse the two estimates (multiplicative, as in the reference)
        n = n_t * n_s
        p = p_t * p_s
        pi = pi_t * pi_s

        n = self._unfold_channels(n, b, d, squeeze_back)
        p = self._unfold_channels(p, b, d, squeeze_back)
        pi = self._unfold_channels(pi, b, d, squeeze_back)
        return n, p, pi
