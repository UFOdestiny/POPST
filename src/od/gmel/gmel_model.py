import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
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

        # 第一层 GAT
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                num_heads=num_heads,
                allow_zero_in_degree=True,
                activation=self.activation,
            )
        )
        # 中间层
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
        # 输出层
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
        # 依次通过各个 GAT 层（注意这里假定最后一层不进行 flatten）
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)

        # 输出层：取多个头的均值
        embeddings = self.gat_layers[-1](g, h).mean(1)
        return embeddings


class GMEL(BaseModel):
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
        参数 T 表示输入的时间步数（即输入的第二个维度）。
        """
        super(GMEL, self).__init__(node_num, input_dim, output_dim)
        # 用于将时间序列信息映射到 X 维特征（适配 GAT 的 in_dim）
        self.mlp_origin = nn.Linear(seq_len, in_dim)
        self.mlp_dest = nn.Linear(seq_len, in_dim)

        self.gat_in = GAT(in_dim, out_dim, num_hidden, num_layers, num_heads)
        self.gat_out = GAT(in_dim, out_dim, num_hidden, num_layers, num_heads)

        # 双线性层，将两个 64 维的节点嵌入融合，输出 1 维
        self.bilinear = nn.Bilinear(out_dim, out_dim, horizon)
        self.out_dim = out_dim

        self.init = False
        self.adj = g
        self.g=None

        

    def forward(self, x, label=None):
        """
        参数 x 的形状为 (B, T, N_src, N_dst)
        参数 g 为图结构信息（要求对应 batched 图，其中每个图的节点数应分别为 N_src 或 N_dst，
        这里假定对源和目的分别采用相同的图结构，或已构造好对应的 batched 图）。
        最终输出形状为 (B, 1, N_src, N_dst)。
        """
        B, T, N_src, N_dst = x.shape
        if not self.init:
            g = build_graph(self.adj)
            self.g = dgl.batch([g] * B).to(device="cuda")
            # self.init = True

        # 对起点（源）分支：沿目的地维度取均值，得到 (B, T, N_src)
        origin_input = x.mean(dim=3)  # shape: (B, T, N_src)
        origin_input = origin_input.permute(0, 2, 1)  # shape: (B, N_src, T)
        # 将时间序列映射到 131 维
        origin_feat = self.mlp_origin(origin_input)  # shape: (B, N_src, 131)
        # 合并 batch 与节点数，输入 GAT
        origin_feat = origin_feat.reshape(B * N_src, -1)  # shape: (B*N_src, 131)
        # 经过 GAT 模块，得到 64 维嵌入（假定 g 已经是 batched graph，节点数对应 B*N_src）
        h_in = self.gat_in(self.g, origin_feat)  # shape: (B*N_src, 64)
        h_in = h_in.reshape(B, N_src, self.out_dim)  # shape: (B, N_src, 64)

        # 对终点（目的）分支：沿起点维度取均值，得到 (B, T, N_dst)
        dest_input = x.mean(dim=2)  # shape: (B, T, N_dst)
        dest_input = dest_input.permute(0, 2, 1)  # shape: (B, N_dst, T)
        dest_feat = self.mlp_dest(dest_input)  # shape: (B, N_dst, 131)
        dest_feat = dest_feat.reshape(B * N_dst, -1)  # shape: (B*N_dst, 131)
        h_out = self.gat_out(self.g, dest_feat)  # shape: (B*N_dst, 64)
        h_out = h_out.reshape(B, N_dst, self.out_dim)  # shape: (B, N_dst, 64)

        # 计算每对起点–终点之间的流量预测
        # 首先扩展维度以便广播，h_in: (B, N_src, 64) -> (B, N_src, N_dst, 64)
        # h_out: (B, N_dst, 64) -> (B, N_src, N_dst, 64)
        h_in_exp = h_in.unsqueeze(2).expand(B, N_src, N_dst, self.out_dim)
        h_out_exp = h_out.unsqueeze(1).expand(B, N_src, N_dst, self.out_dim)

        # 为了使用 bilinear 层，将其展平为二维形式
        h_in_flat = h_in_exp.reshape(B * N_src * N_dst, self.out_dim)
        h_out_flat = h_out_exp.reshape(B * N_src * N_dst, self.out_dim)
        flow = self.bilinear(h_in_flat, h_out_flat)  # shape: (B*N_src*N_dst, 1)
        flow = flow.reshape(B, N_src, N_dst, 1).permute(
            0, 3, 1, 2
        )  # reshape 成 (B, 1, N_src, N_dst)
        return flow
