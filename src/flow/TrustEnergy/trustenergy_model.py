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
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def padding(w_in, w_out, kernel=1, stride=1):
    padding = (w_out - w_in + stride - 1) / 2
    if not padding.is_integer():
        raise ValueError(f"Invalid configuration: padding={padding} is not integer.")
    return int(padding)

class Norm_T(nn.Module):
    def __init__(self, c_in, c_out, feature, seq_len, horizon=3):
        super(Norm_T, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(seq_len, 1),
                                padding=(0, padding(feature, horizon)),
                                bias=True)

    def forward(self, x):  # B, Horizon, N, F
        x = x.permute(0, 2, 1, 3)
        # (B, N, T, f) = x.shape  # B: batch_size; N: input nodes
        loc = self.n_conv(x)
        loc = F.softplus(loc)
        loc = loc.permute(0, 3, 1, 2)

        return loc

class Norm_S(nn.Module):
    def __init__(self, c_in, c_out, feature):
        super(Norm_S, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

    def forward(self, x):  # B, Horizon, N, F
        # x = x.permute(0, 2, 1, 3)
        # (B, N, T, f) = x.shape  # B: batch_size; N: input nodes

        loc = self.n_conv(x)

        loc = F.softplus(loc)

        # loc = loc.permute(0, 2, 1, 3)

        return loc

class DGCN(nn.Module):
    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(DGCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))

        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
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

        x = torch.reshape(x, shape=[self.num_matrices, feature, num_node, input_size, batch_size])
        x = x.permute(1, 4, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[feature, batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        x = x.permute(1, 3, 2, 0)
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', device='cuda'):
        super(TCN, self).__init__()
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
        inv_idx = torch.arange(Xf.size(1) - 1, -1, -1).long().to(
            device=self.device)  # .to(device=self.device).to(device=self.device)
        Xb = Xf.index_select(1, inv_idx)  # inverse the direction of time

        Xf = Xf.permute(0, 2, 3, 1)
        Xb = Xb.permute(0, 2, 3, 1)  # (batch_size, num_nodes, 1, num_timesteps)

        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))  # +
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels, num_features])

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))  # +
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels, num_features])

        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels, num_features]).to(
            device=self.device)  # .to(device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)  # (batch_size, num_timesteps, out_features)

        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=self.device)  # .to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        out = outf + outb
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)
        return out


class mGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv

class mGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        self.gate_conv = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels * 2,
            kernel_size=(kernel_size, 1), padding=(self.padding, 0)
        )
        self.candidate_conv = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels,
            kernel_size=(kernel_size, 1), padding=(self.padding, 0)
        )
        self.out_proj = nn.Conv2d(
            hidden_channels, in_channels,
            kernel_size=1
        )

    def forward(self, x):
        B, T, N, Fe = x.shape
        h = torch.zeros(B, self.hidden_channels, 1, N, device=x.device)  

        outputs = []
        for t in range(T):
            x_t = x[:, t:t+1]  # (B, 1, N, F)
            x_t_ = x_t.permute(0, 3, 1, 2)  # (B, F, 1, N)
            h_prev_ = h if t == 0 else outputs[-1]  # (B, hidden, 1, N)

            gate_input = torch.cat([x_t_, h_prev_], dim=1)  # (B, F+hidden, 1, N)
            gates = self.gate_conv(gate_input)  # (B, 2*hidden, ?, N)

            gates = gates[:, :, 0, :]  # (B, 2*hidden, N)

            r, z = torch.chunk(gates, 2, dim=1)  # (B, hidden, N), (B, hidden, N)
            r, z = torch.sigmoid(r), torch.sigmoid(z)

            r_h_prev = r * h_prev_[:, :, 0, :]  # (B, hidden, N)

            candidate_input = torch.cat([x_t_[:, :, 0, :], r_h_prev], dim=1)  # (B, F+hidden, N)

            candidate_input = candidate_input.unsqueeze(2)  # (B, F+hidden, 1, N)

            h_tilde = torch.tanh(self.candidate_conv(candidate_input))  # (B, hidden, 1, N)
            h_tilde = h_tilde[:, :, 0, :]

            h_t = (1 - z) * h_prev_[:, :, 0, :] + z * h_tilde  # (B, hidden, N)

            h_t = h_t.unsqueeze(2)  # (B, hidden, 1, N)
            outputs.append(h_t)

        h_stack = torch.cat(outputs, dim=2)  # (B, hidden, T, N)
        out = self.out_proj(h_stack)  # (B, in_channels, T, N)
        out = out.permute(0, 2, 3, 1)  # (B, T, N, F)
        return F.relu(out)


class TrustE(BaseModel):
    def __init__(self, A, node_num, hidden_dim_t, hidden_dim_s, rank_t, rank_s,
                 num_timesteps_input, num_timesteps_output, device, input_dim, output_dim, seq_len, min_vec, **args):
        super(TrustE, self).__init__(node_num, input_dim, output_dim, **args)

        self.num_feature = input_dim
        self.seq_len = seq_len

        self.TC1 = TCN(node_num, hidden_dim_t, kernel_size=3).to(device=device)
        self.TC2 = TCN(hidden_dim_t, rank_t, kernel_size=3, activation='linear').to(device=device)
        self.TC3 = TCN(rank_t, hidden_dim_t, kernel_size=3).to(device=device)
        self.TGau = Norm_T(hidden_dim_t, node_num, self.num_feature, self.seq_len, self.horizon).to(device=device)

        self.SC1 = DGCN(num_timesteps_input, hidden_dim_s, 3).to(device=device)
        self.SC2 = DGCN(hidden_dim_s, rank_s, 2, activation='linear').to(device=device)
        self.SC3 = DGCN(rank_s, hidden_dim_s, 2).to(device=device)
        self.SGau = Norm_S(hidden_dim_s, num_timesteps_output, self.num_feature).to(device=device)

        self.A = A
        self.A_q = torch.from_numpy(calculate_random_walk_matrix(self.A).T.astype('float32'))
        self.A_h = torch.from_numpy(calculate_random_walk_matrix(self.A.T).T.astype('float32'))
        self.A_q = self.A_q.to(device=device)
        self.A_h = self.A_h.to(device=device)

        self.mGRU=mGRU(in_channels=1, hidden_channels=16, kernel_size=2)
        
        # embedding_dim = 8
        # node_embedding_dim=16
        # self.hour_embedding = nn.Embedding(24, embedding_dim).to(device="cuda")
        # self.day_embedding = nn.Embedding(7, embedding_dim).to(device="cuda")
        # # self.month_embedding = nn.Embedding(12, embedding_dim).to(device="cuda")
        # self.node_embedding = nn.init.xavier_normal_(nn.Parameter(torch.empty(node_num, node_embedding_dim))).to(device="cuda")

    def forward(self, X, label=None):
        # batch_size, input_len, N, feature
        org_X=X
        X,HDM=X[...,[0]], X[...,1:]
        
        # tod = org_X[:, -1, 0, 1]
        # dow = org_X[:, -1, 0, 2]
        # hour_emb = self.hour_embedding((tod*24).long())
        # day_emb = self.day_embedding(dow.long())
        # # month_emb = self.month_embedding(HDM[:,-1,0,2].long())
        # time_embedding = torch.cat([hour_emb, day_emb], dim=-1) #[64, 48] , month_emb
        # support = torch.softmax(torch.relu(self.node_embedding @ self.node_embedding.T), dim=-1)

        X_t1 = self.TC1(X)
        X_t2 = self.TC2(X_t1)
        X_t3 = self.TC3(X_t2)
        loc_t = self.TGau(X_t3)
        t=self.mGRU(loc_t)

        X_s1 = self.SC1(X, self.A_q, self.A_h)
        X_s2 = self.SC2(X_s1, self.A_q, self.A_h)
        X_s3 = self.SC3(X_s2, self.A_q, self.A_h)
        loc_s = self.SGau(X_s3)
        s=self.mGRU(loc_s)

        # res = loc_s + loc_t
        return s+t
