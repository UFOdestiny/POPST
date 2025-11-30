import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base.model import BaseModel


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    将时间序列分割成patches并进行嵌入
    """
    def __init__(self, input_dim, patch_len, stride, d_model, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # 线性投影层
        self.projection = nn.Linear(patch_len * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            patches: (batch, num_patches, d_model)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 计算patch数量
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # 提取patches
        patches = []
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :]  # (batch, patch_len, input_dim)
            patch = patch.reshape(batch_size, -1)  # (batch, patch_len * input_dim)
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * input_dim)
        
        # 线性投影
        patches = self.projection(patches)  # (batch, num_patches, d_model)
        patches = self.norm(patches)
        patches = self.dropout(patches)
        
        return patches


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class FlattenHead(nn.Module):
    """预测头，将encoder输出映射到预测结果"""
    def __init__(self, d_model, num_patches, horizon, output_dim, dropout=0.1):
        super(FlattenHead, self).__init__()
        
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(num_patches * d_model, d_model)
        self.linear2 = nn.Linear(d_model, horizon * output_dim)
        self.dropout = nn.Dropout(dropout)
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, d_model)
        Returns:
            output: (batch, horizon, output_dim)
        """
        x = self.flatten(x)  # (batch, num_patches * d_model)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)  # (batch, horizon * output_dim)
        x = x.view(-1, self.horizon, self.output_dim)
        return x


class PatchTST(BaseModel):
    """
    PatchTST: A Time Series is Worth 64 Words
    
    Reference: https://arxiv.org/abs/2211.14730
    
    主要特点：
    1. 将时间序列分割成子序列patches
    2. 使用Transformer编码器处理patches
    3. Channel-independent: 每个变量独立处理
    """
    def __init__(
        self,
        patch_len=2,
        stride=1,
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_layers=3,
        dropout=0.1,
        **args
    ):
        super(PatchTST, self).__init__(**args)
        
        # 自动调整patch_len以适应seq_len
        if patch_len > self.seq_len:
            patch_len = max(1, self.seq_len // 2)
        if stride > patch_len:
            stride = max(1, patch_len // 2)
        
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 计算patch数量
        self.num_patches = max(1, (self.seq_len - patch_len) // stride + 1)
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            input_dim=self.input_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1, dropout=dropout)
        
        # Transformer编码器
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 预测头
        self.head = FlattenHead(
            d_model=d_model,
            num_patches=self.num_patches,
            horizon=self.horizon,
            output_dim=self.output_dim,
            dropout=dropout
        )

    def forward(self, x, label=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, node_num, input_dim)
            label: 标签（可选）
            
        Returns:
            output: 预测结果，形状为 (batch, horizon, node_num, output_dim)
        """
        batch_size, seq_len, node_num, input_dim = x.shape
        
        # 重塑输入: (batch * node_num, seq_len, input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, node_num, seq_len, input_dim)
        x = x.view(batch_size * node_num, seq_len, input_dim)
        
        # Patch嵌入: (batch * node_num, num_patches, d_model)
        x = self.patch_embedding(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.encoder(x)  # (batch * node_num, num_patches, d_model)
        
        # 预测头
        x = self.head(x)  # (batch * node_num, horizon, output_dim)
        
        # 重塑输出: (batch, horizon, node_num, output_dim)
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (batch, horizon, node_num, output_dim)
        
        return x


class PatchTSTWithNodeMixing(BaseModel):
    """
    带有节点混合的PatchTST变体
    
    在Channel-independent的基础上增加节点间的信息交互
    """
    def __init__(
        self,
        patch_len=2,
        stride=1,
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_layers=3,
        dropout=0.1,
        node_mixing_layers=1,
        **args
    ):
        super(PatchTSTWithNodeMixing, self).__init__(**args)
        
        # 自动调整patch_len以适应seq_len
        if patch_len > self.seq_len:
            patch_len = max(1, self.seq_len // 2)
        if stride > patch_len:
            stride = max(1, patch_len // 2)
        
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 计算patch数量
        self.num_patches = max(1, (self.seq_len - patch_len) // stride + 1)
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            input_dim=self.input_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1, dropout=dropout)
        
        # Transformer编码器（时序）
        self.temporal_encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 节点混合层
        self.node_mixing = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=node_mixing_layers,
            dropout=dropout
        )
        
        # 预测头
        self.head = FlattenHead(
            d_model=d_model,
            num_patches=self.num_patches,
            horizon=self.horizon,
            output_dim=self.output_dim,
            dropout=dropout
        )

    def forward(self, x, label=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, node_num, input_dim)
            label: 标签（可选）
            
        Returns:
            output: 预测结果，形状为 (batch, horizon, node_num, output_dim)
        """
        batch_size, seq_len, node_num, input_dim = x.shape
        
        # 重塑输入: (batch * node_num, seq_len, input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * node_num, seq_len, input_dim)
        
        # Patch嵌入
        x = self.patch_embedding(x)  # (batch * node_num, num_patches, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 时序Transformer编码
        x = self.temporal_encoder(x)  # (batch * node_num, num_patches, d_model)
        
        # 节点混合
        # 重塑为 (batch * num_patches, node_num, d_model)
        x = x.view(batch_size, node_num, self.num_patches, self.d_model)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, num_patches, node_num, d_model)
        x = x.view(batch_size * self.num_patches, node_num, self.d_model)
        
        x = self.node_mixing(x)  # (batch * num_patches, node_num, d_model)
        
        # 重塑回 (batch * node_num, num_patches, d_model)
        x = x.view(batch_size, self.num_patches, node_num, self.d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * node_num, self.num_patches, self.d_model)
        
        # 预测头
        x = self.head(x)  # (batch * node_num, horizon, output_dim)
        
        # 重塑输出
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)
        
        return x
