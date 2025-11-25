import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base.model import BaseModel
from mamba_ssm import Mamba


class PositionalEncoding(nn.Module):
    """
    可学习的位置编码 + 正弦位置编码的混合
    增强模型对时间位置的感知能力
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # 可学习的位置编码
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # 融合门控
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model) 带位置编码的特征
        """
        T = x.size(1)
        sinusoidal = self.pe[:, :T, :]
        learnable = self.learnable_pe[:, :T, :]
        
        # 门控融合两种位置编码
        combined = torch.cat([sinusoidal.expand_as(x), learnable.expand_as(x)], dim=-1)
        gate = self.gate(combined)
        pe = gate * sinusoidal + (1 - gate) * learnable
        
        return self.dropout(x + pe)


class BidirectionalMamba(nn.Module):
    """
    双向Mamba层：同时捕获前向和后向的时序依赖
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model)
        self.backward_mamba = Mamba(d_model=d_model)
        
        # 融合前向和后向特征
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        # 前向 Mamba
        forward_out = self.forward_mamba(x)
        
        # 后向 Mamba（翻转序列）
        x_reversed = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])
        
        # 融合双向特征
        combined = torch.cat([forward_out, backward_out], dim=-1)
        out = self.fusion(combined)
        
        return out


class AttentionGate(nn.Module):
    """
    注意力门控机制：用于跳跃连接的特征选择
    借鉴 Attention U-Net 的思想，让模型学习关注重要的特征
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # 门控网络
        self.W_g = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model, bias=False)
        self.psi = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, gate, skip):
        """
        Args:
            gate: (B, T, d_model) - 来自解码器的特征（门控信号）
            skip: (B, T, d_model) - 来自编码器的跳跃连接特征
        Returns:
            out: (B, T, d_model) - 加权后的特征
        """
        g = self.W_g(gate)
        x = self.W_x(skip)
        
        # 计算注意力权重
        attention = self.psi(F.relu(g + x))  # (B, T, 1)
        
        # 应用注意力权重
        out = skip * attention
        out = self.norm(out)
        out = self.dropout(out)
        
        return out


class ConvFFN(nn.Module):
    """
    卷积前馈网络：结合局部卷积和全局MLP
    增强局部模式的捕获能力
    """
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expansion
        
        # 深度可分离卷积分支
        self.dwconv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        # MLP 分支
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        # 卷积分支
        conv_out = self.dwconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # MLP 分支
        mlp_out = self.mlp(x)
        
        # 融合
        out = self.norm(conv_out + mlp_out)
        return out


class EncoderBlockV2(nn.Module):
    """
    改进的编码器块：
    1. 双向Mamba捕获双向依赖
    2. 残差连接增强梯度流动
    3. 卷积FFN增强局部建模
    4. Pre-norm结构提高训练稳定性
    """
    def __init__(self, d_model, n_mamba_layers=1, dropout=0.1, use_bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.use_bidirectional = use_bidirectional

        # Pre-norm 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Mamba层（可选双向）
        if use_bidirectional:
            self.mambas = nn.ModuleList([
                BidirectionalMamba(d_model, dropout) for _ in range(n_mamba_layers)
            ])
        else:
            self.mambas = nn.ModuleList([
                Mamba(d_model=d_model) for _ in range(n_mamba_layers)
            ])
        
        # 卷积前馈网络
        self.ffn = ConvFFN(d_model, expansion=4, dropout=dropout)

        # 时间降采样：使用步长=2的卷积
        self.downsample_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 跳跃连接的特征增强
        self.skip_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B*N, T, d_model)
        Returns:
            x_down: (B*N, T//2, d_model)
            x_skip: (B*N, T, d_model) - 增强后的跳跃连接
        """
        # Mamba块 + 残差连接
        residual = x
        x = self.norm1(x)
        for mamba in self.mambas:
            x = mamba(x)
            x = self.dropout(x)
        x = x + residual  # 残差连接
        
        # FFN块 + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual  # 残差连接
        
        # 保存增强后的跳跃连接
        x_skip = self.skip_enhance(x)

        # 时间降采样
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.downsample_conv(x)  # (B*N, d_model, T//2)
        x_down = x.permute(0, 2, 1)  # (B*N, T//2, d_model)

        return x_down, x_skip


class DecoderBlockV2(nn.Module):
    """
    改进的解码器块：
    1. 注意力门控的跳跃连接
    2. 双向Mamba进行特征精化
    3. 多尺度特征融合
    4. 残差连接
    """
    def __init__(self, d_model, n_mamba_layers=1, dropout=0.1, use_bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.use_bidirectional = use_bidirectional

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 时间上采样
        self.upsample_conv = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 注意力门控（用于跳跃连接）
        self.attention_gate = AttentionGate(d_model, dropout)

        # 特征融合：使用更复杂的融合策略
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Mamba层
        if use_bidirectional:
            self.mambas = nn.ModuleList([
                BidirectionalMamba(d_model, dropout) for _ in range(n_mamba_layers)
            ])
        else:
            self.mambas = nn.ModuleList([
                Mamba(d_model=d_model) for _ in range(n_mamba_layers)
            ])

        # 卷积前馈网络
        self.ffn = ConvFFN(d_model, expansion=4, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        """
        Args:
            x: (B*N, T_low, d_model)
            skip: (B*N, T_skip, d_model)
        Returns:
            out: (B*N, T_skip, d_model)
        """
        # 时间上采样
        x = x.permute(0, 2, 1)  # (B*N, d_model, T_low)
        x = self.upsample_conv(x)  # (B*N, d_model, T_up)
        x = x.permute(0, 2, 1)  # (B*N, T_up, d_model)

        # 对齐时间维度
        T_up = x.shape[1]
        T_skip = skip.shape[1]
        if T_up != T_skip:
            if T_up < T_skip:
                x = F.interpolate(
                    x.permute(0, 2, 1), size=T_skip, mode="linear", align_corners=False
                ).permute(0, 2, 1)
            else:
                x = x[:, :T_skip, :]

        # 注意力门控的跳跃连接
        gated_skip = self.attention_gate(x, skip)

        # 特征融合
        fused = torch.cat([x, gated_skip], dim=-1)  # (B*N, T, 2*d_model)
        fused = self.fusion(fused)  # (B*N, T, d_model)

        # Mamba块 + 残差连接
        residual = fused
        out = self.norm1(fused)
        for mamba in self.mambas:
            out = mamba(out)
            out = self.dropout(out)
        out = out + residual

        # FFN块 + 残差连接
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = out + residual

        return out


class MultiScaleBottleneck(nn.Module):
    """
    多尺度瓶颈层：在最低分辨率进行多尺度特征提取
    使用不同大小的感受野捕获不同尺度的模式
    """
    def __init__(self, d_model, n_mamba_layers=2, dropout=0.1):
        super().__init__()
        
        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(d_model // 4),
                nn.GELU(),
            ) for k in [1, 3, 5, 7]
        ])
        
        # 融合多尺度特征
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 双向Mamba层
        self.mambas = nn.ModuleList([
            BidirectionalMamba(d_model, dropout) for _ in range(n_mamba_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B*N, T, d_model)
        Returns:
            out: (B*N, T, d_model)
        """
        # 多尺度卷积
        x_conv = x.permute(0, 2, 1)  # (B*N, d_model, T)
        multi_scale = []
        for conv in self.conv_branches:
            branch_out = conv(x_conv)
            # 确保输出长度一致
            if branch_out.shape[2] != x_conv.shape[2]:
                branch_out = F.interpolate(branch_out, size=x_conv.shape[2], mode='linear', align_corners=False)
            multi_scale.append(branch_out)
        
        # 拼接多尺度特征
        multi_scale = torch.cat(multi_scale, dim=1)  # (B*N, d_model, T)
        multi_scale = multi_scale.permute(0, 2, 1)  # (B*N, T, d_model)
        
        # 融合
        fused = self.fusion(multi_scale)
        
        # 残差连接
        out = x + fused
        
        # 通过双向Mamba进行深度建模
        for mamba in self.mambas:
            residual = out
            out = self.norm(out)
            out = mamba(out)
            out = self.dropout(out)
            out = out + residual
            
        return out


class TemporalProjection(nn.Module):
    """
    改进的时间投影层：从seq_len投影到horizon
    使用多层感知机 + 残差连接
    """
    def __init__(self, seq_len, horizon, d_model, dropout=0.1):
        super().__init__()
        
        # 主投影路径
        self.proj1 = nn.Linear(seq_len, (seq_len + horizon) // 2)
        self.proj2 = nn.Linear((seq_len + horizon) // 2, horizon)
        
        # 直接投影路径（用于残差）
        self.direct_proj = nn.Linear(seq_len, horizon)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B*N, T, d_model) where T = seq_len
        Returns:
            out: (B*N, H, d_model) where H = horizon
        """
        # (B*N, d_model, T)
        x = x.permute(0, 2, 1)
        
        # 主路径
        main = self.proj1(x)
        main = self.act(main)
        main = self.dropout(main)
        main = self.proj2(main)
        
        # 残差路径
        residual = self.direct_proj(x)
        
        # 融合
        out = main + residual
        out = out.permute(0, 2, 1)  # (B*N, H, d_model)
        
        return out


class myMamba2V2(BaseModel):
    """
    改进的 U-Net Mamba 模型 V2
    
    核心改进:
    1. 双向Mamba：同时捕获前向和后向的时序依赖
    2. 注意力门控跳跃连接：自适应选择重要特征
    3. 多尺度瓶颈层：捕获不同尺度的时序模式
    4. 残差连接：增强梯度流动
    5. 位置编码：增强时序位置感知
    6. 卷积FFN：增强局部建模能力
    7. 改进的时间投影：更好的序列到序列映射
    
    输入: (B, T, N, F)
    输出: (B, H, N, F)
    """
    def __init__(
        self,
        d_model,
        feature,
        num_levels=3,
        n_mamba_per_block=1,
        dropout=0.1,
        use_bidirectional=True,
        **args
    ):
        """
        Args:
            d_model: 隐藏维度（推荐64-128）
            feature: 输入特征维度
            num_levels: U-Net层级数（推荐2-4）
            n_mamba_per_block: 每个块的Mamba层数（推荐1-2）
            dropout: Dropout比率
            use_bidirectional: 是否使用双向Mamba
            **args: 其他BaseModel参数
        """
        super(myMamba2V2, self).__init__(**args)
        self.d_model = d_model
        self.feature = feature
        self.num_levels = num_levels
        self.n_mamba_per_block = n_mamba_per_block
        self.dropout_rate = dropout
        self.use_bidirectional = use_bidirectional

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # 编码器堆栈
        self.encoders = nn.ModuleList([
            EncoderBlockV2(
                d_model,
                n_mamba_layers=n_mamba_per_block,
                dropout=dropout,
                use_bidirectional=use_bidirectional
            ) for _ in range(num_levels)
        ])

        # 多尺度瓶颈层
        self.bottleneck = MultiScaleBottleneck(
            d_model,
            n_mamba_layers=n_mamba_per_block + 1,  # 瓶颈处使用更多层
            dropout=dropout
        )

        # 解码器堆栈
        self.decoders = nn.ModuleList([
            DecoderBlockV2(
                d_model,
                n_mamba_layers=n_mamba_per_block,
                dropout=dropout,
                use_bidirectional=use_bidirectional
            ) for _ in range(num_levels)
        ])

        # 时间投影层
        self.time_proj = TemporalProjection(
            self.seq_len, self.horizon, d_model, dropout
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.feature)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, T, N, F)
        Returns:
            output: (B, H, N, F)
        """
        B, T, N, F = x.shape

        # 1. 维度变换: (B, T, N, F) -> (B*N, T, F)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x = x.reshape(B * N, T, F)  # (B*N, T, F)

        # 2. 特征投影
        x = self.input_proj(x)  # (B*N, T, d_model)

        # 3. 添加位置编码
        x = self.pos_encoding(x)  # (B*N, T, d_model)

        # 4. 编码器
        skips = []
        cur = x
        for encoder in self.encoders:
            cur, skip = encoder(cur)
            skips.append(skip)

        # 5. 瓶颈层
        cur = self.bottleneck(cur)

        # 6. 解码器（反向遍历跳跃连接）
        for decoder, skip in zip(self.decoders, reversed(skips)):
            cur = decoder(cur, skip)

        # 7. 对齐到原始时间维度
        T_recon = cur.shape[1]
        if T_recon != T:
            if T_recon < T:
                cur = F.interpolate(
                    cur.permute(0, 2, 1), size=T, mode="linear", align_corners=False
                ).permute(0, 2, 1)
            else:
                cur = cur[:, :T, :]

        # 8. 时间投影
        cur = self.time_proj(cur)  # (B*N, H, d_model)

        # 9. 输出投影
        cur = self.output_proj(cur)  # (B*N, H, F)

        # 10. 恢复维度: (B*N, H, F) -> (B, H, N, F)
        cur = cur.reshape(B, N, self.horizon, F)
        cur = cur.permute(0, 2, 1, 3)  # (B, H, N, F)

        return cur


# 为了向后兼容，保留原始类名
class myMamba2(myMamba2V2):
    """向后兼容的别名"""
    pass
