import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseODModel
from dgl.nn.pytorch import GATConv
import dgl


def build_graph(adjm):
    dst, src = adjm.nonzero()
    d = adjm[adjm.nonzero()]
    g = dgl.graph(([], []))
    g.add_nodes(adjm.shape[0])
    g.add_edges(src, dst, {"d": torch.tensor(d).float().view(-1, 1)})
    return g


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, num_layers, num_heads):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        num_heads = num_heads
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        # first GAT layer
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                num_heads=num_heads,
                allow_zero_in_degree=True,
                activation=self.activation,
            )
        )
        # intermediate layers
        for _ in range(1, self.num_layers):
            self.gat_layers.append(
                GATConv(
                    num_hidden * num_heads,
                    num_hidden,
                    num_heads=num_heads,
                    allow_zero_in_degree=True,
                    activation=self.activation,
                )
            )
        # output layer
        self.gat_layers.append(
            GATConv(
                num_hidden * num_heads,
                out_dim,
                num_heads=num_heads,
                allow_zero_in_degree=True,
                activation=None,
            )
        )

    def forward(self, g, nfeat):
        h = nfeat
        # pass through each GAT layer in turn (assumes the last layer is not flattened here)
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)

        # output layer: average over the multiple heads
        embeddings = self.gat_layers[-1](g, h).mean(1)
        return embeddings


class GMEL(BaseODModel):
    def __init__(
        self,
        node_num,
        input_dim,
        output_dim,
        seq_len,
        horizon,
        g,
        in_dim=64,
        out_dim=64,
        num_hidden=32,
        num_layers=3,
        num_heads=6,
    ):
        """
        The parameter T denotes the number of input time steps (i.e. the second dimension of the input).
        """
        super(GMEL, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)
        # maps the time-series information to X-dim features (matching the GAT in_dim)
        self.mlp_origin = nn.Linear(seq_len, in_dim)
        self.mlp_dest = nn.Linear(seq_len, in_dim)

        self.gat_in = GAT(in_dim, out_dim, num_hidden, num_layers, num_heads)
        self.gat_out = GAT(in_dim, out_dim, num_hidden, num_layers, num_heads)

        # bilinear layer that fuses two 64-dim node embeddings into a horizon-dim output
        self.bilinear = nn.Bilinear(out_dim, out_dim, horizon)
        self.out_dim = out_dim

        self.adj = g
        # Build the single (unbatched) DGL graph once, on CPU. It is replicated
        # and moved to the input device lazily in forward_single, and the
        # batched graph is cached/rebuilt only when the effective batch size B
        # changes (channel-as-batch makes B = B_orig * D, and the final batch
        # may be partial).
        self._base_graph = build_graph(self.adj)
        self._cached_B = None
        self._cached_g = None

    def forward_single(self, x, label=None):
        """
        The parameter x has shape (B, T, N_src, N_dst) where B = B_orig * D
        (channels folded into the batch by BaseODModel).
        The final output has shape (B, horizon, N_src, N_dst).
        """
        B, T, N_src, N_dst = x.shape
        dev = x.device
        # (Re)build the batched graph only when the batch size changes; move it
        # to the input's device (never hardcode "cuda").
        if B != self._cached_B or self._cached_g is None:
            self._cached_g = dgl.batch([self._base_graph] * B).to(dev)
            self._cached_B = B
        self.g = self._cached_g

        # origin (source) branch: average over the destination dimension to get (B, T, N_src)
        origin_input = x.mean(dim=3)  # shape: (B, T, N_src)
        origin_input = origin_input.permute(0, 2, 1)  # shape: (B, N_src, T)
        # map the time series to 131 dims
        origin_feat = self.mlp_origin(origin_input)  # shape: (B, N_src, 131)
        # merge batch and node count, feed into GAT
        origin_feat = origin_feat.reshape(B * N_src, -1)  # shape: (B*N_src, 131)
        # pass through the GAT module to get 64-dim embeddings (assumes g is already a batched graph with node count B*N_src)
        h_in = self.gat_in(self.g, origin_feat)  # shape: (B*N_src, 64)
        h_in = h_in.reshape(B, N_src, self.out_dim)  # shape: (B, N_src, 64)

        # destination branch: average over the origin dimension to get (B, T, N_dst)
        dest_input = x.mean(dim=2)  # shape: (B, T, N_dst)
        dest_input = dest_input.permute(0, 2, 1)  # shape: (B, N_dst, T)
        dest_feat = self.mlp_dest(dest_input)  # shape: (B, N_dst, 131)
        dest_feat = dest_feat.reshape(B * N_dst, -1)  # shape: (B*N_dst, 131)
        h_out = self.gat_out(self.g, dest_feat)  # shape: (B*N_dst, 64)
        h_out = h_out.reshape(B, N_dst, self.out_dim)  # shape: (B, N_dst, 64)

        # compute the flow prediction for every origin-destination pair
        # first expand dims for broadcasting, h_in: (B, N_src, 64) -> (B, N_src, N_dst, 64)
        # h_out: (B, N_dst, 64) -> (B, N_src, N_dst, 64)
        h_in_exp = h_in.unsqueeze(2).expand(B, N_src, N_dst, self.out_dim)
        h_out_exp = h_out.unsqueeze(1).expand(B, N_src, N_dst, self.out_dim)

        # flatten to 2D form so the bilinear layer can be applied
        h_in_flat = h_in_exp.reshape(B * N_src * N_dst, self.out_dim)
        h_out_flat = h_out_exp.reshape(B * N_src * N_dst, self.out_dim)
        flow = self.bilinear(h_in_flat, h_out_flat)  # shape: (B*N_src*N_dst, horizon)
        flow = flow.reshape(B, N_src, N_dst, self.horizon).permute(
            0, 3, 1, 2
        )  # reshape to (B, horizon, N_src, N_dst)
        return flow
