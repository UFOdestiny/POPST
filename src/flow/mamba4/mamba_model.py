"""
Mamba4: Decoupled Spatial-Temporal Mamba (解耦时空Mamba)

核心设计思想：
- 完全解耦时间和空间维度的建模
- 时间维度用纯Mamba (已证明效果好)
- 空间维度用轻量级图网络 (作为可选增强)
- 双流架构：时间流为主，空间流为辅

创新点：
1. Dual-Stream Design: 时间和空间完全分离建模
2. Graph Message Passing: 使用简单的消息传递而非复杂图卷积
3. Stream Fusion Gate: 可学习的双流融合门控
4. Spatial Mamba: 在空间维度也使用Mamba，统一架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class GraphMessagePassing(nn.Module):
    """
    轻量级图消息传递：简单的一跳聚合
    不使用复杂的GCN，而是直接根据邻接矩阵聚合邻居信息
    """

    def __init__(self, d_model, node_num, dropout=0.1, base_adj=None):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num

        # 消息变换
        self.msg_proj = nn.Linear(d_model, d_model)
        # 聚合后的变换
        self.update_proj = nn.Linear(d_model * 2, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 邻接矩阵
        if base_adj is not None:
            # 行归一化
            adj = base_adj.clone().detach().float()
            adj = adj + torch.eye(node_num, device=adj.device, dtype=adj.dtype)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            adj = adj / deg
            self.register_buffer("adj", adj)
        else:
            # 均匀邻接
            adj = torch.ones(node_num, node_num) / node_num
            self.register_buffer("adj", adj)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 消息计算
        msg = self.msg_proj(x)  # (B, T, N, D)

        # 消息聚合 (简单的矩阵乘法)
        msg_flat = msg.reshape(B * T, N, D)  # (B*T, N, D)
        agg = torch.matmul(self.adj, msg_flat)  # (B*T, N, D)
        agg = agg.reshape(B, T, N, D)

        # 更新
        out = torch.cat([x, agg], dim=-1)  # (B, T, N, 2D)
        out = self.update_proj(out)  # (B, T, N, D)
        out = self.dropout(out)

        return self.norm(out + x)


class TemporalMambaBlock(nn.Module):
    """
    时序Mamba块：在时间维度建模
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 转换为 (B*N, T, D) 在时间维度做Mamba
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        residual = x_flat

        x_flat = self.norm(x_flat)
        x_flat = self.mamba(x_flat)
        x_flat = self.dropout(x_flat)
        x_flat = x_flat + residual

        return x_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)


class SpatialMambaBlock(nn.Module):
    """
    空间Mamba块：在空间（节点）维度建模
    将节点序列视为一个序列，用Mamba捕获节点间依赖
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 转换为 (B*T, N, D) 在节点维度做Mamba
        x_flat = x.reshape(B * T, N, D)
        residual = x_flat

        x_flat = self.norm(x_flat)
        x_flat = self.mamba(x_flat)
        x_flat = self.dropout(x_flat)
        x_flat = x_flat + residual

        return x_flat.reshape(B, T, N, D)


class DualStreamFusion(nn.Module):
    """
    双流融合门控：融合时间流和空间流的输出
    可以学习到一方权重为0，相当于关闭一个流
    """

    def __init__(self, d_model, init_spatial_gate=-1.0):
        super().__init__()
        # 融合门控
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # 两个权重
        )
        # 初始化偏向时间流
        nn.init.constant_(self.gate_proj[-1].bias, torch.tensor([1.0, init_spatial_gate]))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_temporal, x_spatial):
        """
        Args:
            x_temporal: (B, T, N, D) 时间流输出
            x_spatial: (B, T, N, D) 空间流输出
        Returns:
            fused: (B, T, N, D)
        """
        # 计算融合权重
        combined = torch.cat([x_temporal, x_spatial], dim=-1)  # (B, T, N, 2D)
        gates = F.softmax(self.gate_proj(combined), dim=-1)  # (B, T, N, 2)

        # 加权融合
        fused = gates[..., 0:1] * x_temporal + gates[..., 1:2] * x_spatial

        return self.norm(fused)


class DecoupledSTBlock(nn.Module):
    """
    解耦时空块：
    1. 时间流：用Mamba建模时序依赖
    2. 空间流：用图消息传递 + 空间Mamba
    3. 融合：双流门控融合
    """

    def __init__(self, d_model, node_num, dropout=0.1, base_adj=None, use_spatial_mamba=True):
        super().__init__()
        self.use_spatial_mamba = use_spatial_mamba

        # 时间流
        self.temporal_mamba = TemporalMambaBlock(d_model, dropout)

        # 空间流
        self.graph_mp = GraphMessagePassing(d_model, node_num, dropout, base_adj)
        if use_spatial_mamba:
            self.spatial_mamba = SpatialMambaBlock(d_model, dropout)

        # 融合
        self.fusion = DualStreamFusion(d_model, init_spatial_gate=-1.0)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        """
        # 时间流
        x_temporal = self.temporal_mamba(x)

        # 空间流
        x_spatial = self.graph_mp(x)
        if self.use_spatial_mamba:
            x_spatial = self.spatial_mamba(x_spatial)

        # 融合
        return self.fusion(x_temporal, x_spatial)


class DecoupledSTMamba(BaseModel):
    """
    Mamba4: 解耦时空Mamba

    架构设计原则：
    1. 双流设计：时间和空间完全分离
    2. 时间流为主：时间Mamba是核心
    3. 空间流可选：通过门控可以关闭空间流
    4. 统一架构：时间和空间都使用Mamba
    """

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        dropout=0.1,
        adj=None,
        use_spatial_mamba=True,
        **args
    ):
        super(DecoupledSTMamba, self).__init__(**args)
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

        # 解耦时空块
        self.blocks = nn.ModuleList([
            DecoupledSTBlock(
                d_model=self.d_model,
                node_num=self.node_num,
                dropout=dropout,
                base_adj=adj,
                use_spatial_mamba=use_spatial_mamba,
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

        # Decoupled ST blocks
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
