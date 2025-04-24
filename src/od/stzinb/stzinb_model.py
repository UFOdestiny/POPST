import math
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
import numpy as np


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(
            np.ones(A.shape[0], dtype=np.float32)
        )  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


class Norm_T(nn.Module):
    def __init__(self, c_in, c_out, seq_len):
        super(Norm_T, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(seq_len, 1), bias=True
        )
        self.out_dim = c_out  # output horizon

    def forward(self, x):  # B, Horizon, N, F
        x = x.permute(0, 2, 1, 3)
        # (B, N, T, f) = x.shape  # B: batch_size; N: input nodes

        loc = self.n_conv(x)

        loc = F.softplus(loc)

        loc = loc.permute(0, 2, 1, 3)

        return loc


class Norm_S(nn.Module):
    def __init__(self, c_in, c_out):
        super(Norm_S, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )

        self.out_dim = c_out  # output horizon

    def forward(self, x):  # B, Horizon, N, F
        # x = x.permute(0, 2, 1, 3)
        # (B, N, T, f) = x.shape  # B: batch_size; N: input nodes

        loc = self.n_conv(
            x
        )  # The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.

        loc = F.softplus(loc)

        # loc = loc.permute(0, 2, 1, 3)
        # scale = scale.permute(0, 1, 3, 2, 4)

        return loc


class mGaussNorm_T(nn.Module):
    def __init__(self, c_in, c_out, feature, seq_len, min_vec):
        super(mGaussNorm_T, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.min_vec = min_vec
        self.n_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(seq_len, 1), bias=True
        )
        self.out_dim = c_out  # output horizon

        self.half = (feature + 1) * feature // 2
        self.full = feature**2
        self.feature = feature
        self.idx_up = torch.triu_indices(self.feature, self.feature)
        self.idx_diag = list(range(self.feature))

        if feature % 2 == 0:
            A, B = feature // 2, (feature + 1)
            self.width = B - A
        else:
            B = (feature + 1) // 2
            self.width = 1
        self.height = seq_len - B + 1

        self.p_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(self.height, self.width),  # 卷积核大小 (height, width)
            bias=True,
        )

    def forward(self, x):  # B, Horizon, N, F
        x = x.permute(0, 2, 1, 3)
        # (B, N, T, f) = x.shape  # B: batch_size; N: input nodes

        loc = self.n_conv(x)
        scale = self.p_conv(x).unsqueeze(2)
        scale = torch.flatten(scale, start_dim=3)

        loc = F.softplus(loc)
        scale = F.softplus(scale)

        loc = loc.permute(0, 2, 1, 3)
        scale = scale.permute(0, 2, 1, 3)

        # return loc, scale
        return loc, sigma_to_matrix(
            sigma=scale, feature=self.feature, index=self.idx_up, min_vec=self.min_vec
        )


class mGaussNorm_S(nn.Module):
    def __init__(self, c_in, c_out, feature, min_vec):
        super(mGaussNorm_S, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.min_vec = min_vec
        self.n_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )

        self.out_dim = c_out  # output horizon

        self.half = (feature + 1) * feature // 2
        self.full = feature**2
        self.feature = feature
        self.idx_up = torch.triu_indices(self.feature, self.feature)
        self.idx_diag = list(range(self.feature))

        if feature % 2 == 0:
            out = feature + 1
            k = feature // 2 + 1

        else:
            out = feature
            k = (feature + 1) // 2

        self.p_conv = nn.Conv2d(
            in_channels=c_in, out_channels=out, kernel_size=(1, k), bias=True
        )

    def forward(self, x):  # B, Horizon, N, F
        # x = x.permute(0, 2, 1, 3)
        # (B, N, T, f) = x.shape  # B: batch_size; N: input nodes

        loc = self.n_conv(x)
        scale = self.p_conv(x).unsqueeze(1)

        loc = F.softplus(loc)
        scale = F.softplus(scale)

        scale = scale.permute(0, 1, 3, 2, 4)
        scale = torch.flatten(scale, start_dim=3)

        # return loc, scale
        return loc, sigma_to_matrix(
            sigma=scale, feature=self.feature, index=self.idx_up, min_vec=self.min_vec
        )


class MDGCN(nn.Module):
    def __init__(self, in_channels, out_channels, orders, activation="relu"):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
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
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size, input_size, num_node, feature = X.shape
        # batch_size = X.shape[0]  # batch_size
        # num_node = X.shape[1]
        # input_size = X.size(2)  # time_length
        # # feature = X.shape[3]

        supports = [A_q, A_h]

        x0 = X.permute(3, 2, 1, 0)  # (num_nodes, num_times, batch_size)
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
        x = x.permute(1, 4, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[feature, batch_size, num_node, input_size * self.num_matrices]
        )
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "selu":
            x = F.selu(x)

        x = x.permute(1, 3, 2, 0)
        return x


class ITCN(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation="relu", device="cuda"
    ):
        super(ITCN, self).__init__()
        # forward dirction temporal convolution
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):

        # batch_size = X.shape[0]
        # seq_len = X.shape[1]
        # Xf = X.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)

        batch_size, seq_len, num_nodes, num_features = X.shape

        Xf = X
        inv_idx = (
            torch.arange(Xf.size(1) - 1, -1, -1).long().to(device=self.device)
        )  # .to(device=self.device).to(device=self.device)
        Xb = Xf.index_select(1, inv_idx)  # inverse the direction of time

        Xf = Xf.permute(0, 2, 3, 1)
        Xb = Xb.permute(0, 2, 3, 1)  # (batch_size, num_nodes, 1, num_timesteps)

        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))  # +
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape(
            [
                batch_size,
                seq_len - self.kernel_size + 1,
                self.out_channels,
                num_features,
            ]
        )

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))  # +
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape(
            [
                batch_size,
                seq_len - self.kernel_size + 1,
                self.out_channels,
                num_features,
            ]
        )

        rec = torch.zeros(
            [batch_size, self.kernel_size - 1, self.out_channels, num_features]
        ).to(
            device=self.device
        )  # .to(device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat(
            (outb, rec), dim=1
        )  # (batch_size, num_timesteps, out_features)

        inv_idx = (
            torch.arange(outb.size(1) - 1, -1, -1).long().to(device=self.device)
        )  # .to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        out = outf + outb
        if self.activation == "relu":
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == "sigmoid":
            out = F.sigmoid(outf) + F.sigmoid(outb)
        return out


class STZINB(BaseModel):
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
        **args
    ):
        super(STZINB, self).__init__(node_num, input_dim, output_dim, **args)

        self.seq_len = seq_len

        self.TC1 = ITCN(node_num, hidden_dim_t, kernel_size=3).to(device=device)
        self.TC2 = ITCN(hidden_dim_t, rank_t, kernel_size=3, activation="linear").to(
            device=device
        )
        self.TC3 = ITCN(rank_t, hidden_dim_t, kernel_size=3).to(device=device)
        self.TGau = Norm_T(hidden_dim_t, node_num, self.seq_len).to(device=device)

        self.SC1 = MDGCN(num_timesteps_input, hidden_dim_s, 3).to(device=device)
        self.SC2 = MDGCN(hidden_dim_s, rank_s, 2, activation="linear").to(device=device)
        self.SC3 = MDGCN(rank_s, hidden_dim_s, 2).to(device=device)
        self.SGau = Norm_S(hidden_dim_s, num_timesteps_output).to(device=device)

        self.A = A
        self.A_q = torch.from_numpy(
            calculate_random_walk_matrix(self.A).T.astype("float32")
        )
        self.A_h = torch.from_numpy(
            calculate_random_walk_matrix(self.A.T).T.astype("float32")
        )
        self.A_q = self.A_q.to(device=device)
        self.A_h = self.A_h.to(device=device)

        self.space_factors = None
        self.temporal_factors = None

    def forward(self, X, label=None):
        # batch_size, input_len, N, feature

        X_t1 = self.TC1(X)
        X_t2 = self.TC2(X_t1)
        X_t3 = self.TC3(X_t2)
        X_1 = self.TGau(X_t3)
        # print(X_1.shape)

        # X=X[:,:,:,0].permute(0,2,1)
        X_s1 = self.SC1(X, self.A_q, self.A_h)
        X_s2 = self.SC2(X_s1, self.A_q, self.A_h)
        X_s3 = self.SC3(X_s2, self.A_q, self.A_h)
        X_2 = self.SGau(X_s3)
        
        # print(X_2.shape)
        # exit()
        res = X_1 + X_2

        return res


def sigma_to_matrix(sigma, feature, index, min_vec):
    z = torch.zeros(*sigma.shape[:3], feature, feature).to(device=sigma.device)
    z[..., index[0], index[1]] = sigma[..., :]
    z[..., index[1], index[0]] = sigma[..., :]

    # z[..., self.idx_diag, self.idx_diag] += 1e-2

    # z = (z + z.transpose(-2, -1)) / 2
    # eigval, eigvec = torch.linalg.eigh(z)
    # adjusted_eigval = torch.clamp(eigval, min=self.min_vec)  # 防止过小的特征值
    # pd_matrix = torch.matmul(eigvec, torch.diag_embed(adjusted_eigval))
    # pd_matrix = torch.matmul(pd_matrix, eigvec.transpose(-2, -1))

    # pd_matrix = torch.matmul(z, z.transpose(-2, -1))
    # pd_matrix[..., self.idx_diag, self.idx_diag] += 1e-4

    # eigval, eigvec = torch.linalg.eigh(z)
    # adjusted_eigval = torch.clamp(eigval, min=self.min_vec)

    # step1 = torch.matmul(eigvec, torch.diag_embed(adjusted_eigval))
    # pd_matrix = torch.matmul(step1, eigvec.transpose(-2, -1))

    # 1. matrix_exp
    # pd_matrix = torch.linalg.matrix_exp(z)

    # 2. Symmetrize and ensure positive definiteness
    eigval, eigvec = torch.linalg.eigh(z)
    clamp_eigval = torch.clamp(eigval, min=min_vec)
    step1 = torch.matmul(eigvec, torch.diag_embed(clamp_eigval))
    pd_matrix = torch.matmul(step1, eigvec.transpose(-2, -1))

    pd_matrix = 0.5 * (pd_matrix + pd_matrix.transpose(-2, -1))
    # print(f"z min:{z.min()}, covariance min:{pd_matrix.min()}")
    return pd_matrix
