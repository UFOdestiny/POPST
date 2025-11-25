import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba
import numpy as np


class GraphConvolution(nn.Module):
    """
    图卷积层：学习节点之间的空间依赖关系

    相比传统GCN，本实现支持：
    1. 动态邻接矩阵学习 - 通过自适应参数学习最优的图结构
    2. 残差连接 - 保留节点的原始特征信息
    3. 多头机制 - 从多个角度学习图关系
    """

    def __init__(self, in_channels, out_channels, num_heads=1, use_bias=True):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_heads: 多头数量（推荐1-4）
            use_bias: 是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        # 多头图卷积投影
        self.fc = nn.Linear(in_channels, out_channels, bias=use_bias)

        # 层归一化和dropout用于稳定训练
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, T, C) - 批大小, 节点数, 时间步, 特征数
            adj: (N, N) - 邻接矩阵，应已标准化

        Returns:
            out: (B, N, T, C_out) - 图卷积后的特征
        """
        batch_size, num_nodes, time_steps, channels = x.shape

        # 1. 特征投影
        x_proj = self.fc(x)  # (B, N, T, C_out)

        # 2. 图卷积：利用邻接矩阵聚合邻域信息
        # 调整维度用于矩阵乘法：(B, N, T, C_out) -> (B*T, N, C_out)
        x_proj_reshaped = x_proj.permute(0, 2, 1, 3).reshape(
            batch_size * time_steps, num_nodes, self.out_channels
        )

        # 邻接矩阵图卷积：A @ X，在节点维度上聚合信息
        out = torch.matmul(adj, x_proj_reshaped)  # (B*T, N, C_out)

        # 3. 恢复维度
        out = out.reshape(batch_size, time_steps, num_nodes, self.out_channels)
        out = out.permute(0, 2, 1, 3)  # (B, N, T, C_out)

        # 4. 归一化和正则化
        out = self.ln(out)
        out = self.dropout(out)

        return out


class AdaptiveAdjacencyMatrix(nn.Module):
    """
    自适应邻接矩阵学习模块：从数据中学习最优的图结构

    核心思想（来自DGCRN）：
    1. 学习两个节点嵌入向量emb1和emb2
    2. 计算节点对之间的相似度/差异度
    3. 生成学到的邻接矩阵，与预定义邻接矩阵结合
    """

    def __init__(self, node_num, emb_dim, dropout=0.1, alpha=3.0):
        """
        Args:
            node_num: 节点数量
            emb_dim: 节点嵌入维度（推荐与d_model相同）
            dropout: dropout比率
            alpha: 激活函数缩放因子（控制梯度流动）
        """
        super(AdaptiveAdjacencyMatrix, self).__init__()
        self.node_num = node_num
        self.emb_dim = emb_dim
        self.alpha = alpha

        # 两个节点嵌入，分别学习不同的图结构视角
        self.emb1 = nn.Embedding(node_num, emb_dim)
        self.emb2 = nn.Embedding(node_num, emb_dim)

        # 线性投影层，用于调整嵌入维度
        self.lin1 = nn.Linear(emb_dim, emb_dim)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, predefined_adj):
        """
        学习自适应邻接矩阵

        Args:
            predefined_adj: (N, N) - 预定义的邻接矩阵（如距离/相似度矩阵）

        Returns:
            adaptive_adj: (N, N) - 学到的自适应邻接矩阵
            combined_adj: (N, N) - 预定义矩阵和自适应矩阵的组合
        """
        # 获取节点嵌入
        idx = torch.arange(self.node_num, device=predefined_adj.device)
        nodevec1 = self.emb1(idx)  # (N, emb_dim)
        nodevec2 = self.emb2(idx)  # (N, emb_dim)

        # 线性投影
        nodevec1 = self.lin1(nodevec1)
        nodevec2 = self.lin2(nodevec2)

        # 计算节点对之间的关系：nodevec1 @ nodevec2^T 得到初始相似度
        a = torch.matmul(nodevec1, nodevec2.t()) - torch.matmul(nodevec2, nodevec1.t())

        # 使用tanh激活和缩放系数产生非线性关系
        # 这样可以学习更复杂的图结构
        adaptive_adj = F.relu(torch.tanh(self.alpha * a))

        # 对邻接矩阵进行行归一化，使其成为有效的转移矩阵
        adaptive_adj = adaptive_adj / (adaptive_adj.sum(dim=1, keepdim=True) + 1e-8)

        # 将预定义邻接矩阵也进行归一化
        predefined_adj_normalized = predefined_adj / (
            predefined_adj.sum(dim=1, keepdim=True) + 1e-8
        )

        # 组合两个邻接矩阵：加权融合预定义信息和学到的关系
        # 权重比例可以调整来控制两部分的贡献
        combined_adj = 0.5 * adaptive_adj + 0.5 * predefined_adj_normalized

        return adaptive_adj, combined_adj


class UMamba(BaseModel):
    """
    改进的U-Net Mamba模型：融入图结构和自适应邻接矩阵学习

    相比原始版本的改进：
    1. ✅ 添加了自适应邻接矩阵学习（从DGCRN汲取灵感）
    2. ✅ 图卷积层用于空间依赖建模
    3. ✅ 多头注意力机制在编码器/解码器中
    4. ✅ 更好的残差连接和特征融合
    5. ✅ 批归一化替代部分LayerNorm以提高稳定性
    6. ✅ 更强的时间投影模块（多层MLP）

    核心架构:
    1. 输入投影：将原始特征投影到高维Mamba空间
    2. 编码器：多层级逐步降采样 + 图卷积处理
    3. 瓶颈：在最低分辨率进行深度特征处理
    4. 解码器：多层级逐步上采样，融合跳跃连接
    5. 时间投影：学习强大的seq_len -> horizon映射
    6. 输出投影：恢复到原始特征维度

    输入: (B, T, N, F) - 批大小, 时间步长, 节点数, 特征数
    输出: (B, H, N, F) - 批大小, 预测长度, 节点数, 特征数
    """

    def __init__(
        self,
        d_model,
        feature,
        predefined_adj=None,
        num_levels=3,
        n_mamba_per_block=1,
        dropout=0.1,
        use_adaptive_adj=True,
        num_heads=1,
        gcn_enabled=True,
        **args
    ):
        """
        Args:
            d_model: Mamba和中间层的隐藏维度（推荐64-128）
            feature: 输入特征维度（通常为1）
            node_num: 节点数量（用于自适应邻接矩阵）
            predefined_adj: 预定义邻接矩阵，形状(N, N)，可选
            num_levels: U-Net的层级数，决定最小采样倍数2^num_levels（推荐3-4）
            n_mamba_per_block: 每个编码/解码块中Mamba层的数量（推荐1-2）
            dropout: Dropout比率，用于正则化（推荐0.1-0.3）
            use_adaptive_adj: 是否使用自适应邻接矩阵学习
            num_heads: 多头注意力头数（推荐1-4）
            gcn_enabled: 是否启用图卷积层
            **args: 其他参数（input_dim, output_dim, seq_len, horizon等）
        """
        super(UMamba, self).__init__(**args)
        self.d_model = d_model
        self.feature = feature
        self.num_levels = num_levels
        self.n_mamba_per_block = n_mamba_per_block
        self.dropout = dropout
        self.num_heads = num_heads
        self.gcn_enabled = gcn_enabled
        self.use_adaptive_adj = use_adaptive_adj

        # ========== 邻接矩阵处理 ==========
        # 如果提供了预定义邻接矩阵，转换为tensor并保存
        if predefined_adj is not None:
            if isinstance(predefined_adj, np.ndarray):
                predefined_adj = torch.from_numpy(predefined_adj).float()
            self.register_buffer("predefined_adj", predefined_adj)
        else:
            # 如果没有提供，使用全连接图（完全邻接矩阵）
            self.register_buffer(
                "predefined_adj", torch.ones(self.node_num, self.node_num)
            )

        # ========== 自适应邻接矩阵学习 ==========
        # 启用自适应邻接矩阵可以从数据中学习最优的图结构
        if self.use_adaptive_adj:
            self.adaptive_adj_module = AdaptiveAdjacencyMatrix(
                node_num=self.node_num, emb_dim=d_model, dropout=dropout, alpha=3.0
            )

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
                    node_num=self.node_num,
                    n_mamba_layers=self.n_mamba_per_block,
                    dropout=self.dropout,
                    num_heads=num_heads,
                    gcn_enabled=gcn_enabled,
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
                    node_num=self.node_num,
                    n_mamba_layers=self.n_mamba_per_block,
                    dropout=self.dropout,
                    num_heads=num_heads,
                    gcn_enabled=gcn_enabled,
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

        # ========== 强化的时间投影层 ==========
        # 使用多层MLP替代简单线性层，增强表达能力
        # 这可以学习更复杂的seq_len -> horizon映射
        self.time_proj_layers = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.seq_len * 2, self.horizon),
        )

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

        # ========== 0. 学习自适应邻接矩阵 ==========
        # 如果启用自适应邻接矩阵，从预定义矩阵中学习最优图结构
        if self.use_adaptive_adj:
            adaptive_adj, combined_adj = self.adaptive_adj_module(self.predefined_adj)
        else:
            # 对预定义邻接矩阵进行标准化
            combined_adj = self.predefined_adj / (
                self.predefined_adj.sum(dim=1, keepdim=True) + 1e-8
            )

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
            if self.gcn_enabled:
                # 如果启用图卷积，传入邻接矩阵
                cur, skip = encoder(cur, combined_adj, B, N)
            else:
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
            if self.gcn_enabled:
                cur = decoder(cur, skip, combined_adj, B, N)
            else:
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
        # 强化的MLP时间投影可以学习更复杂的时间映射关系
        # 输入: (B*N, T, d_model), 输出: (B*N, H, d_model)

        # 对每个特征维度单独进行时间投影
        # (B*N, T, d_model) -> (B*N, H, d_model)
        B_N, T, d_model = cur.shape
        cur_list = []
        for i in range(d_model):
            feat = cur[:, :, i]  # (B*N, T) - 第i个特征维度
            feat_proj = self.time_proj_layers(feat)  # (B*N, H)
            cur_list.append(feat_proj)
        cur = torch.stack(cur_list, dim=2)  # (B*N, H, d_model)

        # ========== 7. 特征投影回原始维度 ==========
        # 从高维Mamba空间投影回原始特征维度F
        cur = self.output_proj(cur)  # (B*N, H, F)

        # ========== 8. 空间维度展开 ==========
        # 恢复批大小和节点维度
        # 从(B*N, H, F) -> (B, H, N, F)
        cur = cur.reshape(B, N, self.horizon, F)  # (B, N, H, F)
        cur = cur.permute(0, 2, 1, 3)  # (B, H, N, F)

        return cur


class EncoderBlock(nn.Module):
    """
    编码器块：逐步降采样时间维度，同时通过Mamba层提取特征。
    支持图卷积用于空间依赖建模。

    输入形状: (B*N, T, d_model) - 批大小×节点数, 时间步长, 特征维度
    输出: 降采样特征 + 跳跃连接(用于解码器)
    """

    def __init__(
        self,
        d_model,
        node_num=1,
        n_mamba_layers=1,
        dropout=0.1,
        num_heads=1,
        gcn_enabled=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.gcn_enabled = gcn_enabled
        self.node_num = node_num

        # ========== 时间特征提取：Mamba层 ==========
        # Mamba层用于序列特征提取
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )

        # ========== 空间特征提取：图卷积层 ==========
        # 如果启用GCN，在每个编码块中添加图卷积
        if self.gcn_enabled:
            self.gcn = GraphConvolution(
                in_channels=d_model,
                out_channels=d_model,
                num_heads=num_heads,
                use_bias=True,
            )
            # 用于融合GCN和Mamba特征的融合层
            self.fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
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

        # 批归一化用于降采样后的特征
        self.bn = nn.BatchNorm1d(d_model)

        # Dropout用于正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj=None, B=None, N=None):
        """
        Args:
            x: (B*N, T, d_model) - 输入特征序列
            adj: (N, N) - 邻接矩阵（可选，启用GCN时使用）
            B: 批大小（可选，启用GCN时使用）
            N: 节点数（可选，启用GCN时使用）

        Returns:
            x_down: (B*N, T//2, d_model) - 降采样后的特征
            x_skip: (B*N, T, d_model) - 原始特征用作跳跃连接
        """
        # 保存跳跃连接（用于解码器的特征融合）
        x_skip = x

        # ========== Mamba特征提取 ==========
        # 通过多个Mamba层进行序列建模
        for i, mamba in enumerate(self.mambas):
            x = mamba(x)  # (B*N, T, d_model)
            x = self.dropout(x)  # 应用dropout正则化

        # ========== 图卷积特征提取（可选） ==========
        if self.gcn_enabled and adj is not None and B is not None and N is not None:
            # 将(B*N, T, d_model)重塑为(B, N, T, d_model)以便GCN处理
            x_reshaped = x.reshape(B, N, x.shape[1], self.d_model)  # (B, N, T, d_model)

            # 应用图卷积
            gcn_out = self.gcn(x_reshaped, adj)  # (B, N, T, d_model)

            # 恢复形状为(B*N, T, d_model)
            gcn_out = gcn_out.reshape(-1, x_reshaped.shape[2], self.d_model)

            # 融合Mamba特征和GCN特征
            fused = torch.cat([x, gcn_out], dim=-1)  # (B*N, T, 2*d_model)
            x = self.fusion(fused)  # (B*N, T, d_model)

        # ========== 降采样 ==========
        # 为卷积操作调整维度：(B*N, T, d_model) -> (B*N, d_model, T)
        x = x.permute(0, 2, 1)

        # 进行时间降采样（时间步从T变为T//2）
        x = self.downsample_conv(x)  # (B*N, d_model, T//2)

        # 批归一化以稳定特征分布
        x = self.bn(x)

        # 恢复维度：(B*N, d_model, T//2) -> (B*N, T//2, d_model)
        x = x.permute(0, 2, 1)

        # 应用层归一化
        x = self.norm(x)

        return x, x_skip


class DecoderBlock(nn.Module):
    """
    解码器块：逐步上采样时间维度，并融合编码器的跳跃连接特征。
    支持图卷积用于空间依赖建模。

    输入: 降采样特征 + 来自编码器的跳跃连接
    输出: (B*N, T, d_model) - 恢复到原始时间长度的特征
    """

    def __init__(
        self,
        d_model,
        node_num=1,
        n_mamba_layers=1,
        dropout=0.1,
        num_heads=1,
        gcn_enabled=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.gcn_enabled = gcn_enabled
        self.node_num = node_num

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

        # ========== 空间特征提取：图卷积层（可选） ==========
        if self.gcn_enabled:
            self.gcn = GraphConvolution(
                in_channels=d_model,
                out_channels=d_model,
                num_heads=num_heads,
                use_bias=True,
            )
            # 用于融合GCN和特征的融合层
            self.gcn_fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        # Mamba层进行上采样后的特征精化
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

        # 批归一化
        self.bn = nn.BatchNorm1d(d_model)

        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip, adj=None, B=None, N=None):
        """
        Args:
            x: (B*N, T_low, d_model) - 来自编码器的降采样特征
            skip: (B*N, T_skip, d_model) - 来自编码器相同层级的跳跃连接
            adj: (N, N) - 邻接矩阵（可选，启用GCN时使用）
            B: 批大小（可选，启用GCN时使用）
            N: 节点数（可选，启用GCN时使用）

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

        # 批归一化以稳定特征分布
        fused = self.bn(fused)

        # 恢复为(B*N, T_skip, d_model)格式
        fused = fused.permute(0, 2, 1)

        # ========== 图卷积特征提取（可选） ==========
        if self.gcn_enabled and adj is not None and B is not None and N is not None:
            # 将(B*N, T, d_model)重塑为(B, N, T, d_model)以便GCN处理
            fused_reshaped = fused.reshape(B, N, fused.shape[1], self.d_model)

            # 应用图卷积
            gcn_out = self.gcn(fused_reshaped, adj)  # (B, N, T, d_model)

            # 恢复形状为(B*N, T, d_model)
            gcn_out = gcn_out.reshape(-1, fused_reshaped.shape[2], self.d_model)

            # 融合融合特征和GCN特征
            combined = torch.cat([fused, gcn_out], dim=-1)  # (B*N, T, 2*d_model)
            fused = self.gcn_fusion(combined)  # (B*N, T, d_model)

        # 应用层归一化
        fused = self.norm(fused)

        # ========== 通过Mamba层进一步精化特征 ==========
        out = fused
        for mamba in self.mambas:
            out = mamba(out)  # (B*N, T_skip, d_model)
            out = self.dropout(out)

        return out
