import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class EncoderBlock(nn.Module):
    """
    编码器块：逐步降采样时间维度，同时通过Mamba层提取特征。

    输入形状: (B*N, T, d_model) - 批大小×节点数, 时间步长, 特征维度
    输出: 降采样特征 + 跳跃连接(用于解码器)
    """

    def __init__(self, d_model, n_mamba_layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers

        # Mamba层用于序列特征提取
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )

        # 时间降采样：使用步长=2的卷积
        # kernel_size=3保证足够的感受野，padding=1防止边界效应
        self.downsample_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=2,
        )

        # 层归一化以稳定梯度流
        self.norm = nn.LayerNorm(d_model)

        # Dropout用于正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B*N, T, d_model) - 输入特征序列

        Returns:
            x_down: (B*N, T//2, d_model) - 降采样后的特征
            x_skip: (B*N, T, d_model) - 原始特征用作跳跃连接
        """
        # 保存跳跃连接（用于解码器的特征融合）
        x_skip = x

        # 通过多个Mamba层进行序列建模
        for i, mamba in enumerate(self.mambas):
            x = mamba(x)  # (B*N, T, d_model)
            x = self.dropout(x)  # 应用dropout正则化

        # 为卷积操作调整维度：(B*N, T, d_model) -> (B*N, d_model, T)
        x = x.permute(0, 2, 1)

        # 进行时间降采样（时间步从T变为T//2）
        x = self.downsample_conv(x)  # (B*N, d_model, T//2)

        # 恢复维度：(B*N, d_model, T//2) -> (B*N, T//2, d_model)
        x = x.permute(0, 2, 1)

        # 应用层归一化稳定特征分布
        x = self.norm(x)

        return x, x_skip


class DecoderBlock(nn.Module):
    """
    解码器块：逐步上采样时间维度，并融合编码器的跳跃连接特征。

    输入: 降采样特征 + 来自编码器的跳跃连接
    输出: (B*N, T, d_model) - 恢复到原始时间长度的特征
    """

    def __init__(self, d_model, n_mamba_layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers

        # 时间上采样：使用转置卷积，将时间维度翻倍
        # kernel_size=4, stride=2, padding=1是标准上采样配置
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 特征融合：在拼接后将通道数从2*d_model投影回d_model
        # 这里使用1×1卷积进行跨通道融合
        self.fusion_conv = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        # Mamba层进行上采样后的特征精化
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        """
        Args:
            x: (B*N, T_low, d_model) - 来自编码器的降采样特征
            skip: (B*N, T_skip, d_model) - 来自编码器相同层级的跳跃连接

        Returns:
            out: (B*N, T_skip, d_model) - 融合后与skip同长度的特征
        """
        # 为转置卷积调整维度：(B*N, T_low, d_model) -> (B*N, d_model, T_low)
        x = x.permute(0, 2, 1)

        # 时间上采样（通常会使T_low翻倍）
        x = self.upsample_conv(x)  # (B*N, d_model, T_up)

        # 恢复维度用于后续处理
        x = x.permute(0, 2, 1)  # (B*N, T_up, d_model)

        # ========== 对齐上采样和跳跃连接的时间维度 ==========
        T_up = x.shape[1]
        T_skip = skip.shape[1]

        if T_up != T_skip:
            # 如果长度不匹配，使用插值进行对齐
            if T_up < T_skip:
                # 上采样不足，使用线性插值扩展
                x = F.interpolate(
                    x.permute(0, 2, 1),  # (B*N, d_model, T_up)
                    size=T_skip,
                    mode="linear",
                    align_corners=False,
                ).permute(
                    0, 2, 1
                )  # (B*N, T_skip, d_model)
            else:
                # 上采样过度，进行裁剪
                x = x[:, :T_skip, :]

        # ========== 特征融合：拼接并投影 ==========
        # 将上采样特征和跳跃连接拼接在通道维度
        fused = torch.cat(
            [x.permute(0, 2, 1), skip.permute(0, 2, 1)],  # 转换为(B*N, C, T)格式
            dim=1,  # 在通道维度拼接
        )  # 结果: (B*N, 2*d_model, T_skip)

        # 通过1×1卷积融合特征
        fused = self.fusion_conv(fused)  # (B*N, d_model, T_skip)

        # 恢复为(B*N, T_skip, d_model)格式
        fused = fused.permute(0, 2, 1)

        # 应用层归一化
        fused = self.norm(fused)

        # ========== 通过Mamba层进一步精化特征 ==========
        out = fused
        for mamba in self.mambas:
            out = mamba(out)  # (B*N, T_skip, d_model)
            out = self.dropout(out)

        return out


class myMamba2(BaseModel):
    """
    U-Net Mamba模型：用于时空序列预测的编码器-解码器架构。

    核心架构:
    1. 输入投影：将原始特征投影到高维Mamba空间
    2. 编码器：多层级逐步降采样，提取多尺度特征
    3. 瓶颈：在最低分辨率进行深度特征处理
    4. 解码器：多层级逐步上采样，融合跳跃连接
    5. 时间投影：将序列长度投影到预测长度
    6. 输出投影：恢复到原始特征维度

    输入: (B, T, N, F) - 批大小, 时间步长, 节点数, 特征数
    输出: (B, H, N, F) - 批大小, 预测长度, 节点数, 特征数
    """

    def __init__(
        self, d_model, feature, num_levels=3, n_mamba_per_block=1, dropout=0.1, **args
    ):
        """
        Args:
            d_model: Mamba和中间层的隐藏维度（推荐64-128）
            feature: 输入特征维度（通常为1）
            num_levels: U-Net的层级数，决定最小采样倍数2^num_levels（推荐3-4）
            n_mamba_per_block: 每个编码/解码块中Mamba层的数量（推荐1-2）
            dropout: Dropout比率，用于正则化（推荐0.1-0.3）
            **args: 其他参数（node_num, input_dim, output_dim, seq_len, horizon等）
        """
        super(myMamba2, self).__init__(**args)
        self.d_model = d_model
        self.feature = feature
        self.num_levels = num_levels
        self.n_mamba_per_block = n_mamba_per_block
        self.dropout = dropout

        # ========== 输入/输出投影 ==========
        # 将输入特征投影到高维Mamba工作空间
        self.input_proj = nn.Linear(self.feature, self.d_model)

        # 从高维特征空间投影回原始特征维度
        self.output_proj = nn.Linear(self.d_model, self.feature)

        # ========== 编码器堆栈 ==========
        # 构建num_levels个编码器块，逐层降采样
        # 最终分辨率为原始的1/(2^num_levels)
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    self.d_model,
                    n_mamba_layers=self.n_mamba_per_block,
                    dropout=self.dropout,
                )
                for _ in range(self.num_levels)
            ]
        )

        # ========== 解码器堆栈 ==========
        # 构建num_levels个解码器块，逐层上采样并融合跳跃连接
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    self.d_model,
                    n_mamba_layers=self.n_mamba_per_block,
                    dropout=self.dropout,
                )
                for _ in range(self.num_levels)
            ]
        )

        # ========== 瓶颈层 ==========
        # 在最低分辨率进行深度的长程依赖建模
        self.bottleneck = nn.ModuleList(
            [Mamba(d_model=self.d_model) for _ in range(self.n_mamba_per_block)]
        )
        self.bottleneck_norm = nn.LayerNorm(self.d_model)
        self.bottleneck_dropout = nn.Dropout(dropout)

        # ========== 时间投影层 ==========
        # 将序列长度从seq_len投影到horizon（预测长度）
        # 这是一个关键的参数化层，学习如何从历史到未来的映射
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, T, N, F)
               B: 批大小
               T: 时间步长（seq_len）
               N: 节点数
               F: 特征维度

        Returns:
            output: (B, H, N, F)
                H: 预测长度（horizon）
        """
        B, T, N, F = x.shape

        # ========== 1. 空间维度折叠 ==========
        # 为了在序列建模时处理所有节点，将批大小和节点数合并
        # 从(B, T, N, F) -> (B*N, T, F)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x = x.reshape(B * N, T, F)  # (B*N, T, F)

        # ========== 2. 特征投影到高维空间 ==========
        # 将特征维度从F投影到d_model
        x = self.input_proj(x)  # (B*N, T, d_model)

        # ========== 3. 编码器：多尺度特征提取 ==========
        # 逐层降采样，同时保存跳跃连接用于解码器
        skips = []
        cur = x

        for level, encoder in enumerate(self.encoders):
            cur, skip = encoder(cur)
            skips.append(skip)
            # 此时cur的时间维度为: T / (2^(level+1))

        # 编码器最后一层后，cur的形状为: (B*N, T/(2^num_levels), d_model)

        # ========== 4. 瓶颈层：深度特征处理 ==========
        # 在最低分辨率上进行深度的上下文建模
        for mamba in self.bottleneck:
            cur = mamba(cur)

        cur = self.bottleneck_norm(cur)
        cur = self.bottleneck_dropout(cur)

        # ========== 5. 解码器：逐步上采样和特征融合 ==========
        # 反向遍历跳跃连接列表，逐层上采样并融合
        for decoder, skip in zip(self.decoders, reversed(skips)):
            cur = decoder(cur, skip)

        # 解码器后，cur应该恢复到(B*N, T, d_model)
        # 但由于数值问题可能存在轻微偏差，进行对齐
        T_recon = cur.shape[1]
        if T_recon != T:
            if T_recon < T:
                # 长度不足，插值扩展
                cur = F.interpolate(
                    cur.permute(0, 2, 1),  # (B*N, d_model, T_recon)
                    size=T,
                    mode="linear",
                    align_corners=False,
                ).permute(
                    0, 2, 1
                )  # (B*N, T, d_model)
            else:
                # 长度过长，裁剪
                cur = cur[:, :T, :]

        # ========== 6. 时间投影：从历史长度到预测长度 ==========
        # 将特征从时间维度T投影到预测长度horizon
        # 这里学习的是从seq_len -> horizon的映射
        cur = cur.permute(0, 2, 1)  # (B*N, d_model, T)
        cur = self.time_proj(cur)  # (B*N, d_model, H)
        cur = cur.permute(0, 2, 1)  # (B*N, H, d_model)

        # ========== 7. 特征投影回原始维度 ==========
        # 从高维Mamba空间投影回原始特征维度F
        cur = self.output_proj(cur)  # (B*N, H, F)

        # ========== 8. 空间维度展开 ==========
        # 恢复批大小和节点维度
        # 从(B*N, H, F) -> (B, H, N, F)
        cur = cur.reshape(B, N, self.horizon, F)  # (B, N, H, F)
        cur = cur.permute(0, 2, 1, 3)  # (B, H, N, F)

        return cur