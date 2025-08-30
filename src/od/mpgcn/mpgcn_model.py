import torch
from torch import nn
import torch.nn.functional as F
from base.model import BaseModel


class GCN(nn.Module):
    def __init__(
        self, K: int, input_dim: int, hidden_dim: int, bias=True, activation=nn.ReLU
    ):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)

    def init_params(self, n_supports: int, b_init=0):
        self.W = nn.Parameter(
            torch.empty(n_supports * self.input_dim, self.hidden_dim),
            requires_grad=True,
        )
        nn.init.xavier_normal_(
            self.W
        )  # sampled from a normal distribution N(0, std^2), also known as Glorot initialization
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, G: torch.Tensor, x: torch.Tensor):
        """
        Batch-wise graph convolution operation on given n support adj matrices
        :param G: support adj matrices - torch.Tensor (K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        """
        assert self.K == G.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum("ij,bjp->bip", [G[k, :, :], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum("bip,pq->biq", [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})"
        )


# class Adj_Processor:
#     def __init__(self, kernel_type: str, K: int):
#         self.kernel_type = kernel_type
#         # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
#         self.K = K if self.kernel_type != "localpool" else 1
#         # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

#     def process(self, flow: torch.Tensor):
#         """
#         Generate adjacency matrices
#         :param flow: batch flow stat - (batch_size, Origin, Destination) torch.Tensor
#         :return: processed adj matrices - (batch_size, K_supports, O, D) torch.Tensor
#         """
#         batch_list = list()

#         for b in range(flow.shape[0]):
#             adj = flow[b, :, :]
#             kernel_list = list()

#             if self.kernel_type in ["localpool", "chebyshev"]:  # spectral
#                 adj_norm = self.symmetric_normalize(adj).to(device="cuda")
#                 if self.kernel_type == "localpool":
#                     localpool = (
#                         torch.eye(adj_norm.shape[0]).to(device="cuda") + adj_norm
#                     )  # same as add self-loop first
#                     kernel_list.append(localpool)

#                 else:  # chebyshev
#                     laplacian_norm = (
#                         torch.eye(adj_norm.shape[0]).to(device="cuda") - adj_norm
#                     )
#                     laplacian_rescaled = self.rescale_laplacian(laplacian_norm)
#                     kernel_list = self.compute_chebyshev_polynomials(
#                         laplacian_rescaled, kernel_list
#                     )

#             elif self.kernel_type == "random_walk_diffusion":  # spatial
#                 # diffuse k steps on transition matrix P
#                 P_forward = self.random_walk_normalize(adj)
#                 kernel_list = self.compute_chebyshev_polynomials(
#                     P_forward.T, kernel_list
#                 )

#             elif self.kernel_type == "dual_random_walk_diffusion":
#                 # diffuse k steps bidirectionally on transition matrix P
#                 P_forward = self.random_walk_normalize(adj)
#                 P_backward = self.random_walk_normalize(adj.T)
#                 forward_series, backward_series = [], []
#                 forward_series = self.compute_chebyshev_polynomials(
#                     P_forward.T, forward_series
#                 )
#                 backward_series = self.compute_chebyshev_polynomials(
#                     P_backward.T, backward_series
#                 )
#                 kernel_list += (
#                     forward_series + backward_series[1:]
#                 )  # 0-order Chebyshev polynomial is same: I

#             else:
#                 raise ValueError(
#                     "Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion]."
#                 )

#             # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
#             kernels = torch.stack(kernel_list, dim=0)
#             batch_list.append(kernels)
#         batch_adj = torch.stack(batch_list, dim=0)
#         return batch_adj

#     @staticmethod
#     def random_walk_normalize(A):  # asymmetric
#         d_inv = torch.pow(A.sum(dim=1), -1)  # OD matrix Ai,j sum on j (axis=1)
#         d_inv[torch.isinf(d_inv)] = 0.0
#         D = torch.diag(d_inv)
#         P = torch.mm(D, A)
#         return P

#     @staticmethod
#     def symmetric_normalize(A):
#         D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
#         D = torch.where(torch.isfinite(D), D, torch.tensor(0.0))
#         A_norm = torch.mm(torch.mm(D, A), D)
#         return A_norm

#     @staticmethod
#     def rescale_laplacian(L):
#         # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
#         try:
#             lambda_ = torch.eig(L)[0][:, 0]  # get the real parts of eigenvalues
#             lambda_max = lambda_.max()  # get the largest eigenvalue
#         except:
#             print(
#                 "Eigen_value calculation didn't converge, using max_eigen_val=2 instead."
#             )
#             lambda_max = 2
#         L_rescaled = (2 / lambda_max) * L - torch.eye(L.shape[0])
#         return L_rescaled

#     def compute_chebyshev_polynomials(self, x, T_k):
#         # compute Chebyshev polynomials up to order k. Return a list of matrices.
#         # print(f"Computing Chebyshev polynomials up to order {self.K}.")
#         for k in range(self.K + 1):
#             if k == 0:
#                 T_k.append(torch.eye(x.shape[0]))
#             elif k == 1:
#                 T_k.append(x)
#             else:
#                 T_k.append(2 * torch.mm(x, T_k[k - 1]) - T_k[k - 2])
#         return T_k

class Adj_Processor:
    def __init__(self, kernel_type: str, K: int):
        self.kernel_type = kernel_type
        self.K = K if kernel_type != "localpool" else 1

    def process(self, flow: torch.Tensor):
        batch_list = []

        for b in range(flow.shape[0]):
            adj = flow[b, :, :]  # shape: (O, D)
            kernel_list = []

            if self.kernel_type in ["localpool", "chebyshev"]:
                adj_norm = self.symmetric_normalize(adj).to(flow.device)
                if self.kernel_type == "localpool":
                    localpool = torch.eye(adj.shape[0], device=flow.device) + adj_norm
                    kernel_list.append(localpool)
                else:
                    laplacian_norm = torch.eye(adj.shape[0], device=flow.device) - adj_norm
                    laplacian_rescaled = self.rescale_laplacian(laplacian_norm)
                    kernel_list = self.compute_chebyshev_polynomials(laplacian_rescaled, kernel_list)

            elif self.kernel_type == "random_walk_diffusion":
                P_forward = self.random_walk_normalize(adj)
                kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)

            elif self.kernel_type == "dual_random_walk_diffusion":
                P_forward = self.random_walk_normalize(adj)
                P_backward = self.random_walk_normalize(adj.T)
                forward_series = self.compute_chebyshev_polynomials(P_forward.T, [])
                backward_series = self.compute_chebyshev_polynomials(P_backward.T, [])
                kernel_list += forward_series + backward_series[1:]

            else:
                raise ValueError(f"Invalid kernel_type: {self.kernel_type}")

            kernels = torch.stack(kernel_list, dim=0)  # shape: (K_supports, O, D)
            batch_list.append(kernels)

        batch_adj = torch.stack(batch_list, dim=0)  # shape: (B, K_supports, O, D)
        return batch_adj.to(device="cuda")

    @staticmethod
    def random_walk_normalize(A):
        row_sum = A.sum(dim=1)
        d_inv = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))
        D = torch.diag(d_inv)
        P = torch.mm(D, A)
        P[torch.isnan(P)] = 0.0
        return P

    @staticmethod
    def symmetric_normalize(A):
        row_sum = A.sum(dim=1)
        d_inv_sqrt = torch.where(row_sum > 0, torch.pow(row_sum, -0.5), torch.zeros_like(row_sum))
        D = torch.diag(d_inv_sqrt)
        A_norm = torch.mm(torch.mm(D, A), D)
        A_norm[torch.isnan(A_norm)] = 0.0
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        try:
            lambda_ = torch.linalg.eigvals(L).real
            lambda_max = lambda_.max().clamp(min=1e-5).item()
        except Exception as e:
            print("Eigenvalue calculation failed:", e)
            lambda_max = 2.0
        I = torch.eye(L.shape[0], device=L.device)
        return (2.0 / lambda_max) * L - I

    def compute_chebyshev_polynomials(self, x, T_k):
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0], device=x.device))
            elif k == 1:
                T_k.append(x)
            else:
                T_next = 2 * torch.mm(x, T_k[k - 1]) - T_k[k - 2]
                T_next[torch.isnan(T_next)] = 0.0
                T_k.append(T_next)
        return T_k



class BDGCN(nn.Module):  # 2DGCN: handling both static and dynamic graph input
    def __init__(
        self, K: int, input_dim: int, hidden_dim: int, use_bias=True, activation=nn.ReLU
    ):
        super(BDGCN, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(
            torch.empty(self.input_dim * (self.K**2), self.hidden_dim),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return

    def forward(self, X: torch.Tensor, G):
        feat_set = list()
        if type(G) == torch.Tensor:  # static graph input: (K, N, N)
            assert self.K == G.shape[-3]
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum("bncl,nm->bmcl", X, G[o, :, :])
                    mode_2_prod = torch.einsum("bmcl,cd->bmdl", mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        elif (
            type(G) == tuple
        ):  # dynamic graph input: ((batch, K, N, N), (batch, K, N, N))
            assert (len(G) == 2) & (self.K == G[0].shape[-3] == G[1].shape[-3])
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum("bncl,bnm->bmcl", X, G[0][:, o, :, :])
                    mode_2_prod = torch.einsum(
                        "bmcl,bcd->bmdl", mode_1_prod, G[1][:, d, :, :]
                    )
                    feat_set.append(mode_2_prod)
        else:
            raise NotImplementedError

        _2D_feat = torch.cat(feat_set, dim=-1)
        mode_3_prod = torch.einsum("bmdk,kh->bmdh", _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H


class MPGCN(BaseModel):
    def __init__(
        self,
        M: int,
        K: int,
        input_dim: int,
        output_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        gcn_hidden_dim: int,
        gcn_num_layers: int,
        num_nodes: int,
        user_bias: bool,
        G: list,
        kernel_type,
        cheby_order,
        activation=nn.ReLU,
        **args,
    ):
        super(MPGCN, self).__init__(
            node_num=num_nodes, input_dim=input_dim, output_dim=output_dim
        )
        self.M = M  # input graphs
        self.K = K  # chebyshev order
        self.G = G
        self.G_tensor = None

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.gcn_num_layers = gcn_num_layers

        # initiate a branch of (LSTM, 2DGCN, FC) for each graph input
        self.branch_models = nn.ModuleList()
        for m in range(self.M):
            branch = nn.ModuleDict()
            branch["temporal"] = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
            )
            branch["spatial"] = nn.ModuleList()
            for n in range(gcn_num_layers):
                cur_input_dim = lstm_hidden_dim if n == 0 else gcn_hidden_dim
                branch["spatial"].append(
                    BDGCN(
                        K=K,
                        input_dim=cur_input_dim,
                        hidden_dim=gcn_hidden_dim,
                        use_bias=user_bias,
                        activation=activation,
                    )
                )
            branch["fc"] = nn.Sequential(
                nn.Linear(
                    in_features=gcn_hidden_dim, out_features=input_dim, bias=True
                ),
                nn.ReLU(),
            )
            self.branch_models.append(branch)

            self.adj_preprocessor = Adj_Processor(kernel_type, cheby_order)

    def init_hidden_list(self, batch_size: int):  # for LSTM initialization
        hidden_list = list()
        for m in range(self.M):
            weight = next(self.parameters()).data
            hidden = (
                weight.new_zeros(
                    self.lstm_num_layers,
                    batch_size * (self.num_nodes**2),
                    self.lstm_hidden_dim,
                ),
                weight.new_zeros(
                    self.lstm_num_layers,
                    batch_size * (self.num_nodes**2),
                    self.lstm_hidden_dim,
                ),
            )
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, x_seq, o, d):
        x_seq = x_seq.unsqueeze(4)  # (batch, seq, N, N, input_dim)

        if self.G_tensor is None:
            self.G = self.adj_preprocessor.process(
                torch.from_numpy(self.G).float().unsqueeze(dim=0)
            ).squeeze(dim=0).to(device="cuda")
            self.G_tensor = 1

        G_list = [
            self.G,
            (
                self.adj_preprocessor.process(o).to(device="cuda"),
                self.adj_preprocessor.process(d).to(device="cuda"),
            ),
        ]

        batch_size, seq_len, N1, N2, i = x_seq.shape
        assert N1 == N2 == self.num_nodes

        hidden_list = self.init_hidden_list(batch_size)

        # flatten nodes for LSTM
        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(
            batch_size * (self.num_nodes**2), seq_len, i
        )

        branch_out = []

        max_batch = 1024
        for m in range(self.M):
            outputs_chunks = []

            B_total = lstm_in.size(0)
            for start in range(0, B_total, max_batch):
                end = min(start + max_batch, B_total)
                chunk = lstm_in[start:end].contiguous()  # [chunk_batch, seq_len, input_size]

                # hidden 按 chunk 初始化
                h = torch.zeros(self.lstm_num_layers, chunk.size(0), self.lstm_hidden_dim, device=chunk.device)
                c = torch.zeros(self.lstm_num_layers, chunk.size(0), self.lstm_hidden_dim, device=chunk.device)

                lstm_out_chunk, (h, c) = self.branch_models[m]["temporal"](chunk, (h, c))
                outputs_chunks.append(lstm_out_chunk)

            lstm_out = torch.cat(outputs_chunks, dim=0)

            # reshape for GCN
            gcn_in = lstm_out[:, -1, :].reshape(batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim)

            for n in range(self.gcn_num_layers):
                gcn_in = self.branch_models[m]["spatial"][n](gcn_in, G_list[m])

            fc_out = self.branch_models[m]["fc"](gcn_in)
            branch_out.append(fc_out)

        # ensemble
        ensemble_out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
        res = ensemble_out.permute(0, 3, 1, 2)
        return res

    # def forward(self, x_seq: torch.Tensor, o, d):  # , G_list: list
    #     """
    #     :param x_seq: (batch, seq, O, D, 1)
    #     :param G_list: static graph (K, N, N); dynamic OD graph tuple ((batch, K, N, N), (batch, K, N, N))
    #     :return:
    #     """
    #     x_seq = x_seq.unsqueeze(4)

    #     if self.G_tensor is None:
    #         self.G = self.adj_preprocessor.process(
    #             torch.from_numpy(self.G).float().unsqueeze(dim=0)
    #         ).squeeze(dim=0)
    #         self.G.to(device="cuda")
    #         self.G_tensor = 1

    #     G_list = [
    #         self.G,
    #         (
    #             self.adj_preprocessor.process(o).to(device="cuda"),
    #             self.adj_preprocessor.process(d).to(device="cuda"),
    #         ),
    #     ]


    #     assert (len(x_seq.shape) == 5) & (
    #         self.num_nodes == x_seq.shape[2] == x_seq.shape[3]
    #     )
    #     assert len(G_list) == self.M
    #     batch_size, seq_len, _, _, i = x_seq.shape
    #     hidden_list = self.init_hidden_list(batch_size)

    #     lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(
    #         batch_size * (self.num_nodes**2), seq_len, i
    #     )
    #     branch_out = list()
    #     for m in range(self.M):
    #         lstm_out, hidden_list[m] = self.branch_models[m]["temporal"](
    #             lstm_in, hidden_list[m]
    #         )

    #         gcn_in = lstm_out[:, -1, :].reshape(
    #             batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim
    #         )

    #         for n in range(self.gcn_num_layers):
    #             # print(G_list[m][0].isnan().any())
    #             gcn_in = self.branch_models[m]["spatial"][n](gcn_in, G_list[m])

    #         fc_out = self.branch_models[m]["fc"](gcn_in)
    #         branch_out.append(fc_out)

    #     # ensemble
    #     ensemble_out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
    #     res = ensemble_out.permute(0, 3, 1, 2)
    #     return res  # match dim for single-step pred
