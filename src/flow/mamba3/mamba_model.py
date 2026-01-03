"""
Mamba3: Mamba + Learnable Graph Attention

核心设计思想：
- 纯Mamba效果好，所以Mamba为主干网络
- 添加可学习的图结构（Adaptive Graph）来捕获空间依赖
- 使用零初始化门控(Zero-Init Gate)保证初始时等价于纯Mamba
- Graph部分使用简单的消息传递机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class AdaptiveGraphConv(nn.Module):
    """
    可学习的自适应图卷积
    使用节点嵌入学习邻接矩阵
    """
    def __init__(self, d_model, node_num, embed_dim=16):
        super().__init__()
        self.node_num = node_num
        self.embed_dim = embed_dim
        
        # 节点嵌入用于学习图结构
        self.node_embed1 = nn.Parameter(torch.randn(node_num, embed_dim))
        self.node_embed2 = nn.Parameter(torch.randn(node_num, embed_dim))
        
        # 消息传递的线性变换
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        
        # 计算自适应邻接矩阵: (N, N)
        adj = F.softmax(F.relu(self.node_embed1 @ self.node_embed2.T), dim=-1)
        
        # 消息传递: (B, T, N, D)
        x_flat = x.reshape(B * T, N, D)  # (BT, N, D)
        out = torch.matmul(adj, x_flat)  # (BT, N, D)
        out = self.fc(out)
        
        return out.reshape(B, T, N, D)


class MambaGraphBlock(nn.Module):
    """
    Mamba + Graph 的基础块
    先时序Mamba，再空间图卷积（带门控）
    """
    def __init__(self, d_model, node_num, dropout=0.1, embed_dim=16):
        super().__init__()
        
        # 1. Temporal Mamba (主力)
        self.mamba = Mamba(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Spatial Graph Conv (辅助)
        self.graph_conv = AdaptiveGraphConv(d_model, node_num, embed_dim)
        self.norm2 = nn.LayerNorm(d_model)
        # 零初始化门控：初始时完全关闭Graph，保证不差于纯Mamba
        self.graph_gate = nn.Parameter(torch.tensor(0.0))
        
        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # 1. 时序Mamba
        residual = x
        x_norm = self.norm1(x)
        x_flat = x_norm.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_mamba = self.mamba(x_flat)
        x_mamba = x_mamba.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = residual + self.dropout(x_mamba)

        # 2. 空间图卷积 (带门控)
        residual = x
        x_norm = self.norm2(x)
        x_graph = self.graph_conv(x_norm)
        x = residual + self.graph_gate * self.dropout(x_graph)
        
        # 3. FFN
        residual = x
        x_norm = self.norm3(x)
        x_ffn = self.ffn(x_norm)
        x = residual + self.dropout(x_ffn)

        return x


class MambaGraph(BaseModel):
    """
    Mamba3: Mamba + Learnable Graph Attention
    """

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        dropout=0.1,
        embed_dim=16,
        **args
    ):
        super(MambaGraph, self).__init__(**args)
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

        # Mamba + Graph Blocks
        self.blocks = nn.ModuleList([
            MambaGraphBlock(
                d_model=self.d_model,
                node_num=self.node_num,
                dropout=dropout,
                embed_dim=embed_dim,
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

        # Mamba + Graph blocks
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
