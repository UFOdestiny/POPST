"""
Mamba5: Bidirectional Mamba with Optional Light Graph Guidance
优化版本: 核心强调纯Mamba的时序建模，图信息作为可选的轻量级增强

设计要点:
1. 双向Mamba: 同时捕获前向和后向的时序依赖
2. 深层残差Mamba堆叠: 更深的网络 + 强残差连接
3. 可学习图门控: 模型可以自动关闭图信息退化为纯Mamba
4. Pre-norm设计: 更稳定的训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class GraphGate(nn.Module):
    """可学习图门控：让模型决定使用多少图信息，初始偏向纯Mamba"""

    def __init__(self, d_model, init_gate=-3.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self):
        return torch.sigmoid(self.gate)


class LightGraphMixer(nn.Module):
    """轻量级图混合: 简单的一跳消息传递，可通过门控关闭"""

    def __init__(self, node_num, embed_dim=16, base_adj=None):
        super().__init__()
        self.node_num = node_num

        # 可学习节点嵌入（用于自适应邻接矩阵）
        self.node_embed = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)

        if base_adj is not None:
            # 归一化base_adj
            adj = base_adj.clone().detach().float()
            adj = adj + torch.eye(node_num, device=adj.device, dtype=adj.dtype)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            adj = adj / deg
            self.register_buffer("base_adj", adj)
        else:
            self.base_adj = None

    def forward(self):
        # 自适应邻接矩阵
        adaptive_adj = torch.mm(self.node_embed, self.node_embed.t())
        adaptive_adj = F.softmax(adaptive_adj, dim=-1)

        if self.base_adj is not None:
            # 融合基础图和自适应图
            adj = 0.5 * self.base_adj + 0.5 * adaptive_adj
        else:
            adj = adaptive_adj
        return adj


class BidirectionalMambaBlock(nn.Module):
    """双向Mamba块: 同时建模前向和后向时序依赖"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model)
        self.mamba_bwd = Mamba(d_model=d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D)"""
        residual = x
        x = self.norm(x)

        # 前向Mamba
        x_fwd = self.mamba_fwd(x)

        # 后向Mamba (翻转 -> 处理 -> 翻转回来)
        x_bwd = torch.flip(x, dims=[1])
        x_bwd = self.mamba_bwd(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])

        # 融合双向信息
        x_cat = torch.cat([x_fwd, x_bwd], dim=-1)
        x = self.fusion(x_cat)
        x = self.dropout(x)

        return x + residual


class DeepMambaBlock(nn.Module):
    """深层Mamba块: 单向Mamba + 前馈层，Pre-norm设计"""

    def __init__(self, d_model, dropout=0.1, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, T, D)"""
        # Mamba with pre-norm
        x = x + self.mamba(self.norm1(x))
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class UNetMamba(BaseModel):
    """
    Mamba5 优化版: 双向深层Mamba + 可选图增强

    架构:
    - 输入投影 + 可选轻量图混合
    - 双向Mamba层 (捕获双向时序)
    - 深层Mamba堆叠 (增强表达能力)
    - 输出投影
    """

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        adj=None,
        graph_embed_dim=16,
        dropout=0.1,
        use_bidirectional=True,
        ffn_expand=2,
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature
        self.use_bidirectional = use_bidirectional

        # Input projection
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

        # 可学习图门控 (初始化为偏向关闭)
        self.graph_gate = GraphGate(d_model, init_gate=-3.0)

        # 轻量级图混合器
        self.graph_mixer = LightGraphMixer(
            node_num=self.node_num,
            embed_dim=graph_embed_dim,
            base_adj=adj,
        )

        # 双向Mamba层 (如果启用)
        if use_bidirectional:
            self.bidirectional_block = BidirectionalMambaBlock(d_model, dropout)

        # 深层Mamba堆叠
        self.mamba_blocks = nn.ModuleList([
            DeepMambaBlock(d_model, dropout, expand=ffn_expand)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def apply_graph_mixing(self, x, gate_value):
        """应用可门控的图混合"""
        if gate_value < 0.01:  # 门控值很小时跳过计算
            return x

        adj = self.graph_mixer()
        # x: (B, T, N, D)
        graph_x = torch.einsum("ij,btjd->btid", adj, x)
        return x + gate_value * (graph_x - x)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, T, N, d_model)
        x = self.input_norm(x)

        # 可选图混合 (通过门控控制)
        gate_value = self.graph_gate()
        x = self.apply_graph_mixing(x, gate_value)

        # Reshape for temporal processing: (B*N, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)

        # 保存用于残差
        x_input = x

        # 双向Mamba (如果启用)
        if self.use_bidirectional:
            x = self.bidirectional_block(x)

        # 深层Mamba堆叠
        for block in self.mamba_blocks:
            x = block(x)

        # 全局残差连接
        x = x + x_input

        # Output projection
        x = self.output_norm(x)
        x = x.permute(0, 2, 1)  # (B*N, D, T)
        x = self.time_proj(x)  # (B*N, D, H)
        x = x.permute(0, 2, 1)  # (B*N, H, D)
        x = self.output_proj(x)  # (B*N, H, F)

        # Reshape back to (B, H, N, F)
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
