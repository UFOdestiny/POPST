"""
Mamba7: Adaptive Graph-Gated Deep Mamba
优化版本: 自适应门控决定图信息使用程度，深层Mamba作为核心

设计要点:
1. 自适应图门控: 可学习的门控让模型自动决定是否使用图信息
2. 深层Mamba堆叠: 更深的Mamba网络，增强时序建模能力
3. 渐进式图融合: 不同层使用不同权重的图信息
4. Mixture of Experts风格: 选择性使用图增强或纯Mamba
5. Pre-norm + 强残差: 稳定训练，保证梯度流动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class AdaptiveGraphGate(nn.Module):
    """自适应图门控: 基于输入内容决定使用多少图信息"""

    def __init__(self, d_model, init_bias=-2.0):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        # 初始化偏置为负值，让模型初期更依赖纯Mamba
        nn.init.constant_(self.gate_net[-1].bias, init_bias)

    def forward(self, x):
        """
        x: (B, T, N, D)
        return: (B, T, N, 1) 门控值在[0, 1]之间
        """
        # 使用时间维度的平均来计算门控
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, N, D)
        gate = torch.sigmoid(self.gate_net(x_mean))  # (B, 1, N, 1)
        return gate.expand(-1, x.size(1), -1, -1)  # (B, T, N, 1)


class LightGraphConv(nn.Module):
    """轻量级图卷积: 简单的消息传递"""

    def __init__(self, node_num, d_model, embed_dim=16, base_adj=None, dropout=0.1):
        super().__init__()
        self.node_num = node_num
        self.d_model = d_model

        # 节点嵌入用于自适应邻接矩阵
        self.node_embed = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)

        # 消息投影
        self.msg_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if base_adj is not None:
            adj = base_adj.clone().detach().float()
            # 添加自环并归一化
            adj = adj + torch.eye(node_num, device=adj.device, dtype=adj.dtype)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            adj = adj / deg
            self.register_buffer("base_adj", adj)
        else:
            self.base_adj = None

    def forward(self, x):
        """
        x: (B, T, N, D)
        return: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 计算自适应邻接矩阵
        adaptive_adj = torch.mm(self.node_embed, self.node_embed.t())
        adaptive_adj = F.softmax(adaptive_adj, dim=-1)

        if self.base_adj is not None:
            adj = 0.7 * self.base_adj + 0.3 * adaptive_adj
        else:
            adj = adaptive_adj

        # 消息传递
        x_flat = x.reshape(B * T, N, D)  # (B*T, N, D)
        msg = self.msg_proj(x_flat)  # (B*T, N, D)
        agg = torch.matmul(adj, msg)  # (B*T, N, D)
        agg = agg.reshape(B, T, N, D)

        return self.norm(self.dropout(agg) + x)


class MambaBlock(nn.Module):
    """标准Mamba块: Pre-norm + 残差"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaWithFFN(nn.Module):
    """Mamba + FFN块: 增强表达能力"""

    def __init__(self, d_model, dropout=0.1, ffn_expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GraphGatedMambaLayer(nn.Module):
    """图门控Mamba层: 结合图信息和纯Mamba"""

    def __init__(self, d_model, node_num, embed_dim=16, base_adj=None, dropout=0.1, gate_init=-2.0):
        super().__init__()

        # 图处理分支
        self.graph_gate = AdaptiveGraphGate(d_model, init_bias=gate_init)
        self.graph_conv = LightGraphConv(node_num, d_model, embed_dim, base_adj, dropout)

        # Mamba分支
        self.mamba = MambaWithFFN(d_model, dropout)

    def forward(self, x):
        """
        x: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # 计算图门控值
        gate = self.graph_gate(x)  # (B, T, N, 1)

        # 图处理 (只有当gate > 阈值时才真正计算)
        if gate.mean() > 0.01:
            x_graph = self.graph_conv(x)
            x = x + gate * (x_graph - x)

        # Mamba处理 (在时间维度)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        x_flat = self.mamba(x_flat)
        x = x_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)

        return x


class DeepMambaStack(nn.Module):
    """深层Mamba堆叠: 纯时序建模"""

    def __init__(self, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaWithFFN(d_model, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """x: (B, T, D)"""
        for layer in self.layers:
            x = layer(x)
        return x


class UNetMamba(BaseModel):
    """
    Mamba7 优化版: 自适应图门控 + 深层Mamba

    架构:
    - 输入投影
    - 图门控Mamba层 (可选使用图信息)
    - 深层纯Mamba堆叠
    - 输出投影

    特点:
    - 模型可以自动学习是否需要图信息
    - 初始化偏向纯Mamba，图信息作为可选增强
    - 深层Mamba保证强时序建模能力
    """

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        adj=None,
        graph_embed_dim=16,
        dropout=0.1,
        num_graph_layers=1,
        gate_init=-2.0,
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature

        # Input projection
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 图门控Mamba层 (带图信息的层，数量较少)
        self.graph_mamba_layers = nn.ModuleList([
            GraphGatedMambaLayer(
                d_model=d_model,
                node_num=self.node_num,
                embed_dim=graph_embed_dim,
                base_adj=adj,
                dropout=dropout,
                gate_init=gate_init - 0.5 * i  # 越深的层越偏向纯Mamba
            )
            for i in range(num_graph_layers)
        ])

        # 深层纯Mamba堆叠
        pure_mamba_layers = max(1, num_layers - num_graph_layers)
        self.deep_mamba = DeepMambaStack(d_model, pure_mamba_layers, dropout)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, T, N, d_model)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # 保存输入用于残差
        x_input = x

        # 图门控Mamba层
        for layer in self.graph_mamba_layers:
            x = layer(x)

        # 转换为时序处理格式
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)  # (B*N, T, D)

        # 深层纯Mamba
        x = self.deep_mamba(x)

        # Reshape回来
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, D)

        # 全局残差
        x = x + x_input

        # 转换并输出
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)  # (B*N, T, D)
        x = self.output_norm(x)
        x = x.permute(0, 2, 1)  # (B*N, D, T)
        x = self.time_proj(x)  # (B*N, D, H)
        x = x.permute(0, 2, 1)  # (B*N, H, D)
        x = self.output_proj(x)  # (B*N, H, F)

        # Reshape back: (B, H, N, F)
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
