import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class GraphConv(nn.Module):
    """
    图卷积层：支持预定义图和自适应图的混合传播
    """

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 融合多个图信息后的线性变换
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        """
        Args:
            x: (B, N, C) - 节点特征
            adj_list: list of (N, N) 邻接矩阵列表
        Returns:
            out: (B, N, C) - 图卷积后的特征
        """
        # 聚合多个图的信息
        out = x
        for adj in adj_list:
            # (B, N, C) @ (N, N) -> (B, N, C)
            out = out + torch.einsum("bnc,nm->bmc", x, adj)

        out = self.fc(out)
        out = self.dropout(out)
        return out


class SpatialGraphConv(nn.Module):
    """
    空间图卷积：结合预定义图和自适应图进行空间信息聚合
    
    支持:
    1. 预定义邻接矩阵（静态空间关系）
    2. 自适应邻接矩阵（数据驱动的动态关系）
    """

    def __init__(self, d_model, node_num, dropout=0.1, use_adaptive=True):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num
        self.use_adaptive = use_adaptive

        # 图卷积的权重
        self.weights_pool = nn.Parameter(torch.randn(d_model, d_model))
        self.bias_pool = nn.Parameter(torch.zeros(d_model))

        if use_adaptive:
            # 自适应图学习的节点嵌入
            self.node_emb1 = nn.Parameter(torch.randn(node_num, 16))
            self.node_emb2 = nn.Parameter(torch.randn(node_num, 16))

        self.fc = nn.Linear(d_model * 2 if use_adaptive else d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, predefined_adj):
        """
        Args:
            x: (B, N, d_model) - 节点特征
            predefined_adj: list of (N, N) - 预定义邻接矩阵列表 [adj, adj_T]
        Returns:
            out: (B, N, d_model)
        """
        B, N, C = x.shape

        # ========== 1. 预定义图卷积 ==========
        # 使用预定义邻接矩阵进行空间聚合
        out_pre = torch.zeros_like(x)
        for adj in predefined_adj:
            # adj: (N, N), x: (B, N, C)
            # einsum: 对每个batch中的节点进行图卷积
            out_pre = out_pre + torch.einsum("nm,bmc->bnc", adj, x)

        # 线性变换
        out_pre = torch.matmul(out_pre, self.weights_pool) + self.bias_pool

        if self.use_adaptive:
            # ========== 2. 自适应图卷积 ==========
            # 通过节点嵌入学习自适应邻接矩阵
            # (N, 16) @ (16, N) -> (N, N)
            adaptive_adj = F.softmax(
                F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=-1
            )

            # 自适应图卷积
            out_adp = torch.einsum("nm,bmc->bnc", adaptive_adj, x)
            out_adp = torch.matmul(out_adp, self.weights_pool) + self.bias_pool

            # ========== 3. 融合预定义图和自适应图 ==========
            out = torch.cat([out_pre, out_adp], dim=-1)
            out = self.fc(out)
        else:
            out = out_pre

        # 残差连接和归一化
        out = self.norm(out + x)
        out = self.dropout(out)

        return out


class GraphMambaBlock(nn.Module):
    """
    图增强的Mamba块：结合时序Mamba和空间图卷积
    
    处理流程:
    1. 空间图卷积：捕获节点间的空间依赖
    2. 时序Mamba：捕获时间维度上的长程依赖
    3. 特征融合：结合空间和时序信息
    """

    def __init__(self, d_model, node_num, n_mamba_layers=1, dropout=0.1, use_adaptive=True):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num

        # 空间图卷积
        self.spatial_conv = SpatialGraphConv(
            d_model, node_num, dropout, use_adaptive
        )

        # 时序Mamba层
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )

        # 时空融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        """
        Args:
            x: (B*N, T, d_model) 或需要reshape
            adj_list: 邻接矩阵列表
        Returns:
            out: 同输入形状
        """
        # 输入形状: (B*N, T, d_model)
        BN, T, C = x.shape
        B = BN // self.node_num
        N = self.node_num

        # ========== 1. 空间图卷积 ==========
        # 重塑为 (B, T, N, C) 以进行空间操作
        x_spatial = x.view(B, N, T, C).permute(0, 2, 1, 3)  # (B, T, N, C)

        # 对每个时间步应用空间图卷积
        spatial_out = []
        for t in range(T):
            x_t = x_spatial[:, t, :, :]  # (B, N, C)
            out_t = self.spatial_conv(x_t, adj_list)  # (B, N, C)
            spatial_out.append(out_t)

        spatial_out = torch.stack(spatial_out, dim=1)  # (B, T, N, C)
        spatial_out = spatial_out.permute(0, 2, 1, 3).reshape(BN, T, C)  # (B*N, T, C)

        # ========== 2. 时序Mamba ==========
        temporal_out = x
        for mamba in self.mambas:
            temporal_out = mamba(temporal_out)
            temporal_out = self.dropout(temporal_out)

        # ========== 3. 时空融合 ==========
        fused = torch.cat([spatial_out, temporal_out], dim=-1)  # (B*N, T, 2*C)
        out = self.fusion(fused)  # (B*N, T, C)

        # 残差连接
        out = self.norm(out + x)

        return out


class GraphEncoderBlock(nn.Module):
    """
    图增强编码器块：使用图卷积增强的Mamba进行编码
    """

    def __init__(self, d_model, node_num, n_mamba_layers=1, dropout=0.1, use_adaptive=True):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num

        # 图增强Mamba块
        self.graph_mamba = GraphMambaBlock(
            d_model, node_num, n_mamba_layers, dropout, use_adaptive
        )

        # 时间降采样
        self.downsample_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=2,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        """
        Args:
            x: (B*N, T, d_model)
            adj_list: 邻接矩阵列表

        Returns:
            x_down: (B*N, T//2, d_model)
            x_skip: (B*N, T, d_model)
        """
        # 保存跳跃连接
        x_skip = x

        # 图增强Mamba处理
        x = self.graph_mamba(x, adj_list)

        # 时间降采样
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.downsample_conv(x)  # (B*N, d_model, T//2)
        x = x.permute(0, 2, 1)  # (B*N, T//2, d_model)

        x = self.norm(x)

        return x, x_skip


class GraphDecoderBlock(nn.Module):
    """
    图增强解码器块：使用图卷积增强的Mamba进行解码
    """

    def __init__(self, d_model, node_num, n_mamba_layers=1, dropout=0.1, use_adaptive=True):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num

        # 时间上采样
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 特征融合
        self.fusion_conv = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        # 图增强Mamba块
        self.graph_mamba = GraphMambaBlock(
            d_model, node_num, n_mamba_layers, dropout, use_adaptive
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip, adj_list):
        """
        Args:
            x: (B*N, T_low, d_model)
            skip: (B*N, T_skip, d_model)
            adj_list: 邻接矩阵列表

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
                    x.permute(0, 2, 1),
                    size=T_skip,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            else:
                x = x[:, :T_skip, :]

        # 特征融合
        fused = torch.cat(
            [x.permute(0, 2, 1), skip.permute(0, 2, 1)],
            dim=1,
        )
        fused = self.fusion_conv(fused)
        fused = fused.permute(0, 2, 1)
        fused = self.norm(fused)

        # 图增强Mamba处理
        out = self.graph_mamba(fused, adj_list)

        return out


class GraphBottleneck(nn.Module):
    """
    图增强瓶颈层：在最低分辨率进行深度时空建模
    """

    def __init__(self, d_model, node_num, n_mamba_layers=2, dropout=0.1, use_adaptive=True):
        super().__init__()
        self.d_model = d_model
        self.node_num = node_num

        # 多个图增强Mamba块
        self.blocks = nn.ModuleList([
            GraphMambaBlock(d_model, node_num, 1, dropout, use_adaptive)
            for _ in range(n_mamba_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list):
        """
        Args:
            x: (B*N, T, d_model)
            adj_list: 邻接矩阵列表

        Returns:
            out: (B*N, T, d_model)
        """
        for block in self.blocks:
            x = block(x, adj_list)

        x = self.norm(x)
        x = self.dropout(x)

        return x


class GraphUMamba(BaseModel):
    """
    图增强的U-Net Mamba模型：结合图神经网络和Mamba进行时空预测
    
    核心改进:
    1. 空间图卷积：在每个编码/解码块中加入图卷积，捕获空间依赖
    2. 自适应图学习：数据驱动的动态图结构学习
    3. 时空融合：在每个层级上融合空间和时序信息
    
    输入: (B, T, N, F) - 批大小, 时间步长, 节点数, 特征数
    输出: (B, H, N, F) - 批大小, 预测长度, 节点数, 特征数
    """

    def __init__(
        self,
        d_model,
        feature,
        predefined_adj,  # 预定义邻接矩阵列表
        num_levels=3,
        n_mamba_per_block=1,
        dropout=0.1,
        use_adaptive=True,  # 是否使用自适应图
        **args
    ):
        """
        Args:
            d_model: Mamba隐藏维度
            feature: 输入特征维度
            predefined_adj: 预定义邻接矩阵列表 [adj, adj_T]
            num_levels: U-Net层级数
            n_mamba_per_block: 每块Mamba层数
            dropout: Dropout比率
            use_adaptive: 是否使用自适应图学习
        """
        super(GraphUMamba, self).__init__(**args)
        self.d_model = d_model
        self.feature = feature
        self.num_levels = num_levels
        self.n_mamba_per_block = n_mamba_per_block
        self.dropout = dropout
        self.use_adaptive = use_adaptive

        # 注册预定义邻接矩阵为buffer（不参与梯度更新但会随模型移动到GPU）
        for i, adj in enumerate(predefined_adj):
            self.register_buffer(f"adj_{i}", adj)
        self.num_adj = len(predefined_adj)

        # 输入/输出投影
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.feature)

        # 编码器堆栈
        self.encoders = nn.ModuleList([
            GraphEncoderBlock(
                self.d_model,
                self.node_num,
                n_mamba_layers=self.n_mamba_per_block,
                dropout=self.dropout,
                use_adaptive=self.use_adaptive,
            )
            for _ in range(self.num_levels)
        ])

        # 解码器堆栈
        self.decoders = nn.ModuleList([
            GraphDecoderBlock(
                self.d_model,
                self.node_num,
                n_mamba_layers=self.n_mamba_per_block,
                dropout=self.dropout,
                use_adaptive=self.use_adaptive,
            )
            for _ in range(self.num_levels)
        ])

        # 瓶颈层
        self.bottleneck = GraphBottleneck(
            self.d_model,
            self.node_num,
            n_mamba_layers=self.n_mamba_per_block,
            dropout=self.dropout,
            use_adaptive=self.use_adaptive,
        )

        # 时间投影
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

    def get_adj_list(self):
        """获取邻接矩阵列表"""
        return [getattr(self, f"adj_{i}") for i in range(self.num_adj)]

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, T, N, F)

        Returns:
            output: (B, H, N, F)
        """
        B, T, N, F = x.shape
        adj_list = self.get_adj_list()

        # 1. 空间维度折叠: (B, T, N, F) -> (B*N, T, F)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x = x.reshape(B * N, T, F)  # (B*N, T, F)

        # 2. 特征投影
        x = self.input_proj(x)  # (B*N, T, d_model)

        # 3. 编码器
        skips = []
        cur = x
        for encoder in self.encoders:
            cur, skip = encoder(cur, adj_list)
            skips.append(skip)

        # 4. 瓶颈层
        cur = self.bottleneck(cur, adj_list)

        # 5. 解码器
        for decoder, skip in zip(self.decoders, reversed(skips)):
            cur = decoder(cur, skip, adj_list)

        # 对齐时间维度
        T_recon = cur.shape[1]
        if T_recon != T:
            if T_recon < T:
                cur = F.interpolate(
                    cur.permute(0, 2, 1),
                    size=T,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            else:
                cur = cur[:, :T, :]

        # 6. 时间投影
        cur = cur.permute(0, 2, 1)  # (B*N, d_model, T)
        cur = self.time_proj(cur)  # (B*N, d_model, H)
        cur = cur.permute(0, 2, 1)  # (B*N, H, d_model)

        # 7. 特征投影回原始维度
        cur = self.output_proj(cur)  # (B*N, H, F)

        # 8. 恢复空间维度
        cur = cur.reshape(B, N, self.horizon, F)  # (B, N, H, F)
        cur = cur.permute(0, 2, 1, 3)  # (B, H, N, F)

        return cur
