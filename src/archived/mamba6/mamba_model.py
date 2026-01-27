"""
Mamba6: Hierarchical Pure Mamba with Multi-Scale Temporal Modeling
优化版本: 移除复杂的UNet上下采样，采用多尺度纯Mamba架构

设计要点:
1. 多尺度时间建模: 不同层关注不同时间尺度
2. 深层纯Mamba: 更深的Mamba堆叠，强调时序建模
3. 层级残差: 每一层都有残差连接到输入
4. 无图结构: 完全依赖Mamba的时序建模能力
5. 可选的时间卷积: 用于捕获局部时间模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class TemporalConvBlock(nn.Module):
    """局部时间卷积: 捕获短期时间模式"""

    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D)"""
        residual = x
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T, D)
        x = self.dropout(x)
        return self.norm(x + residual)


class MambaWithFFN(nn.Module):
    """Mamba + FFN块: Pre-norm设计，增强表达能力"""

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
        # Mamba
        x = x + self.mamba(self.norm1(x))
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class MultiScaleMambaBlock(nn.Module):
    """多尺度Mamba块: 捕获不同时间尺度的依赖"""

    def __init__(self, d_model, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales

        # 每个尺度一个Mamba
        self.scale_mambas = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(num_scales)
        ])

        # 尺度融合
        self.fusion = nn.Linear(d_model * num_scales, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D)"""
        B, T, D = x.shape
        residual = x

        scale_outputs = []
        for i, mamba in enumerate(self.scale_mambas):
            scale = 2 ** i  # 1, 2, 4...

            if scale == 1:
                # 原始尺度
                out = mamba(x)
            else:
                # 下采样 -> Mamba -> 上采样
                if T >= scale:
                    # 简单的stride采样
                    x_down = x[:, ::scale, :]  # (B, T//scale, D)
                    out_down = mamba(x_down)
                    # 上采样回原始长度
                    out = F.interpolate(
                        out_down.permute(0, 2, 1),
                        size=T,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)
                else:
                    out = mamba(x)

            scale_outputs.append(out)

        # 融合多尺度输出
        x = torch.cat(scale_outputs, dim=-1)  # (B, T, D*num_scales)
        x = self.fusion(x)  # (B, T, D)
        x = self.dropout(x)

        return self.norm(x + residual)


class HierarchicalMambaStage(nn.Module):
    """层级Mamba阶段: 多个Mamba块 + 残差连接"""

    def __init__(self, d_model, num_blocks, dropout=0.1, ffn_expand=2, use_temporal_conv=True):
        super().__init__()
        self.use_temporal_conv = use_temporal_conv

        if use_temporal_conv:
            self.temporal_conv = TemporalConvBlock(d_model, kernel_size=3, dropout=dropout)

        self.blocks = nn.ModuleList([
            MambaWithFFN(d_model, dropout, ffn_expand)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """x: (B, T, D)"""
        if self.use_temporal_conv:
            x = self.temporal_conv(x)

        for block in self.blocks:
            x = block(x)

        return x


class UNetMamba(BaseModel):
    """
    Mamba6 优化版: 层级多尺度纯Mamba架构

    架构:
    - 输入投影 + LayerNorm
    - 多个层级Mamba阶段 (每个阶段多个Mamba块)
    - 多尺度Mamba融合 (可选)
    - 层级残差连接
    - 输出投影
    """

    def __init__(
        self,
        d_model,
        num_layers,
        sample_factor,  # 改为控制多尺度数量
        feature,
        dropout=0.1,
        ffn_expand=2,
        use_multiscale=True,
        use_temporal_conv=True,
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature
        self.use_multiscale = use_multiscale

        # Input projection
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 层级Mamba阶段
        blocks_per_stage = max(1, num_layers // 2)  # 每个阶段的块数
        num_stages = max(1, (num_layers + blocks_per_stage - 1) // blocks_per_stage)

        self.stages = nn.ModuleList([
            HierarchicalMambaStage(
                d_model=d_model,
                num_blocks=blocks_per_stage,
                dropout=dropout,
                ffn_expand=ffn_expand,
                use_temporal_conv=use_temporal_conv and (i == 0)  # 只在第一阶段用时间卷积
            )
            for i in range(num_stages)
        ])

        # 多尺度Mamba (可选)
        if use_multiscale:
            self.multiscale_block = MultiScaleMambaBlock(
                d_model=d_model,
                num_scales=min(sample_factor, 3),  # 最多3个尺度
                dropout=dropout
            )

        # 最终Mamba层
        self.final_mamba = Mamba(d_model=d_model)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # Reshape: (B*N, T, F)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # 保存输入用于层级残差
        x_input = x

        # 层级Mamba阶段
        for stage in self.stages:
            x = stage(x)

        # 多尺度融合 (如果启用)
        if self.use_multiscale:
            x = self.multiscale_block(x)

        # 层级残差
        x = x + x_input

        # 最终Mamba精修
        x = self.final_mamba(self.final_norm(x)) + x

        # Output projection
        x = self.output_norm(x)
        x = x.permute(0, 2, 1)  # (B*N, D, T)
        x = self.time_proj(x)  # (B*N, D, H)
        x = x.permute(0, 2, 1)  # (B*N, H, D)
        x = self.output_proj(x)  # (B*N, H, F)

        # Reshape back: (B, H, N, F)
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
