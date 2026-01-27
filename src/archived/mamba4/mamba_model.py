"""
Mamba4: Mamba + Chebyshev Graph Convolution

核心设计思想：
- 纯Mamba效果好，所以Mamba为主干网络
- 使用Chebyshev多项式图卷积捕获多阶邻居信息
- 使用零初始化门控(Zero-Init Gate)保证初始时等价于纯Mamba
- 可选择使用预定义邻接矩阵或自适应学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class ChebGraphConv(nn.Module):
    """
    Chebyshev多项式图卷积
    捕获K阶邻居信息
    """
    def __init__(self, d_model, node_num, K=2, adj=None):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num
        self.K = K
        
        # 处理邻接矩阵
        if adj is not None:
            # 计算归一化拉普拉斯矩阵
            adj = adj.clone().detach().float()
            adj = adj + torch.eye(node_num, device=adj.device)
            deg = adj.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
            deg_inv_sqrt = torch.diag(deg_inv_sqrt)
            lap = torch.eye(node_num, device=adj.device) - deg_inv_sqrt @ adj @ deg_inv_sqrt
            # 缩放到 [-1, 1]
            lambda_max = 2.0
            lap_scaled = (2.0 / lambda_max) * lap - torch.eye(node_num, device=adj.device)
            self.register_buffer('lap', lap_scaled)
        else:
            self.register_buffer('lap', torch.zeros(node_num, node_num))
        
        # 每个阶的权重
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(d_model, d_model))
            for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.zeros(d_model))
        
        # 初始化权重
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        
        # Chebyshev递推
        L = self.lap
        Tx_0 = x_flat
        out = torch.matmul(Tx_0, self.weights[0])
        
        if self.K > 1:
            Tx_1 = torch.matmul(L, Tx_0)
            out = out + torch.matmul(Tx_1, self.weights[1])
            
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(L, Tx_1) - Tx_0
                out = out + torch.matmul(Tx_2, self.weights[k])
                Tx_0, Tx_1 = Tx_1, Tx_2
        
        out = out + self.bias
        return out.reshape(B, T, N, D)


class MambaChebBlock(nn.Module):
    """
    Mamba + Chebyshev GCN 的基础块
    """
    def __init__(self, d_model, node_num, dropout=0.1, K=2, adj=None):
        super().__init__()
        
        # 1. Temporal Mamba (主力)
        self.mamba = Mamba(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Spatial ChebGCN (辅助)
        self.gcn = ChebGraphConv(d_model, node_num, K=K, adj=adj)
        self.norm2 = nn.LayerNorm(d_model)
        # 零初始化门控
        self.gcn_gate = nn.Parameter(torch.tensor(0.0))
        
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

        # 2. 空间ChebGCN (带门控)
        residual = x
        x_norm = self.norm2(x)
        x_gcn = self.gcn(x_norm)
        x = residual + self.gcn_gate * self.dropout(x_gcn)
        
        # 3. FFN
        residual = x
        x_norm = self.norm3(x)
        x_ffn = self.ffn(x_norm)
        x = residual + self.dropout(x_ffn)

        return x


class MambaCheb(BaseModel):
    """
    Mamba4: Mamba + Chebyshev Graph Convolution
    """

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        dropout=0.1,
        K=2,
        adj=None,
        **args
    ):
        super(MambaCheb, self).__init__(**args)
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

        # Mamba + ChebGCN Blocks
        self.blocks = nn.ModuleList([
            MambaChebBlock(
                d_model=self.d_model,
                node_num=self.node_num,
                dropout=dropout,
                K=K,
                adj=adj,
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

        # Mamba + ChebGCN blocks
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
