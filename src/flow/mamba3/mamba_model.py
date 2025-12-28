"""
Mamba3: Graph-Guided Mamba (图引导的Mamba)

核心设计思想：
- 既然纯Mamba效果好，说明时序建模已经足够强
- 图信息应该作为"辅助引导"，而非"主导"
- 使用图结构来调制节点间的信息流，而非强制聚合

创新点：
1. Graph-Guided Selection: 用图结构调制Mamba的选择性机制
2. Sparse Node Attention: 只在top-k相邻节点间做轻量级交互
3. Adaptive Graph Gating: 可学习的门控，让模型决定是否使用图信息
4. Residual-Dominant Design: 残差连接为主，保证模型可以退化为纯Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class AdaptiveGraphGate(nn.Module):
    """
    自适应图门控：让模型决定每个时间步使用多少图信息
    门控值可以学习到接近0，此时模型退化为纯Mamba
    """

    def __init__(self, d_model, init_gate=-2.0):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        # 初始化为小值，让模型初期更依赖Mamba
        nn.init.constant_(self.gate_proj[-1].bias, init_gate)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        Returns:
            gate: (B, T, N, 1) 范围 [0, 1]
        """
        return torch.sigmoid(self.gate_proj(x))


class SparseGraphAttention(nn.Module):
    """
    稀疏图注意力：只在top-k邻居间做注意力
    不是替代Mamba，而是为节点提供局部上下文
    """

    def __init__(self, node_num, d_model, top_k=5, num_heads=4, dropout=0.1, base_adj=None):
        super().__init__()
        self.node_num = node_num
        self.d_model = d_model
        self.top_k = min(top_k, node_num - 1)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 节点嵌入用于计算邻居权重
        self.node_embed = nn.Parameter(torch.randn(node_num, d_model) * 0.02)

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # 预计算top-k邻居 (如果有base_adj)
        if base_adj is not None:
            self.register_buffer("base_adj", base_adj.clone().detach().float())
            # 计算每个节点的top-k邻居索引
            _, topk_indices = torch.topk(base_adj, self.top_k, dim=-1)
            self.register_buffer("neighbor_indices", topk_indices)
        else:
            self.base_adj = None
            self.neighbor_indices = None

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 计算Q, K, V
        Q = self.q_proj(x)  # (B, T, N, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, H, T, N, d)
        K = K.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # 如果有预计算的邻居，使用稀疏注意力
        if self.neighbor_indices is not None:
            # 只在邻居间计算注意力
            # neighbor_indices: (N, top_k)
            neighbor_idx = self.neighbor_indices  # (N, K)

            # 收集邻居的K和V: (B, H, T, N, K, d)
            K_neighbors = K[:, :, :, neighbor_idx, :]  # (B, H, T, N, K, d)
            V_neighbors = V[:, :, :, neighbor_idx, :]

            # 计算注意力分数: (B, H, T, N, K)
            attn = torch.einsum('bhtnd,bhtnkd->bhtnk', Q, K_neighbors) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # 加权聚合: (B, H, T, N, d)
            out = torch.einsum('bhtnk,bhtnkd->bhtnd', attn, V_neighbors)
        else:
            # 全局注意力（fallback）
            attn = torch.einsum('bhtmd,bhtnd->bhtmn', Q, K) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.einsum('bhtmn,bhtnd->bhtmd', attn, V)

        # Reshape back
        out = out.permute(0, 2, 3, 1, 4).reshape(B, T, N, D)
        out = self.out_proj(out)

        return out


class MambaTemporalBlock(nn.Module):
    """
    时序Mamba块：纯时序建模，不涉及空间
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B*N, T, D)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class GraphGuidedMambaBlock(nn.Module):
    """
    图引导的Mamba块：
    1. 先用Mamba捕获时序特征（主力）
    2. 用稀疏图注意力提供空间上下文（辅助）
    3. 门控融合，可以自动关闭图信息
    """

    def __init__(self, d_model, node_num, top_k=5, num_heads=4, dropout=0.1, base_adj=None):
        super().__init__()

        # 时序Mamba (主力)
        self.temporal_mamba = MambaTemporalBlock(d_model, dropout)

        # 稀疏图注意力 (辅助)
        self.graph_attn = SparseGraphAttention(
            node_num=node_num,
            d_model=d_model,
            top_k=top_k,
            num_heads=num_heads,
            dropout=dropout,
            base_adj=base_adj,
        )

        # 自适应门控
        self.gate = AdaptiveGraphGate(d_model, init_gate=-2.0)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 1. 时序Mamba (主力)
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        x_temporal = self.temporal_mamba(x_temporal)
        x_temporal = x_temporal.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)

        # 2. 稀疏图注意力 (辅助)
        x_graph = self.graph_attn(x_temporal)  # (B, T, N, D)

        # 3. 门控融合
        gate = self.gate(x_temporal)  # (B, T, N, 1)
        out = x_temporal + gate * (x_graph - x_temporal)

        return self.norm(out)


class GraphGuidedMamba(BaseModel):
    """
    Mamba3: 图引导的Mamba

    架构设计原则：
    1. Mamba为主：时序建模完全由Mamba负责
    2. 图为辅助：图信息只提供空间上下文，不强制聚合
    3. 可退化性：通过门控可以退化为纯Mamba
    4. 稀疏性：只考虑top-k邻居，避免噪声
    """

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        top_k=5,
        num_heads=4,
        dropout=0.1,
        adj=None,
        **args
    ):
        super(GraphGuidedMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )

        # 图引导Mamba块
        self.blocks = nn.ModuleList([
            GraphGuidedMambaBlock(
                d_model=self.d_model,
                node_num=self.node_num,
                top_k=top_k,
                num_heads=num_heads,
                dropout=dropout,
                base_adj=adj,
            )
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, T, N, D)

        # Skip connection from input
        x_skip = x

        # Graph-Guided Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Global skip connection
        x = x + x_skip

        # Time projection: (B, T, N, D) -> (B, H, N, D)
        x = x.permute(0, 2, 3, 1)  # (B, N, D, T)
        x = self.time_proj(x)      # (B, N, D, H)
        x = x.permute(0, 3, 1, 2)  # (B, H, N, D)

        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)  # (B, H, N, F)

        return x
