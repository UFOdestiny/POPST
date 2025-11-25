import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.model import BaseModel
from mamba_ssm import Mamba
from torch.autograd import Variable


class GCNLayer(nn.Module):
    """
    图卷积层 - 受DGCRN启发的GCN层
    用于在节点维度上进行消息传递
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: (B*N, T, d_model) - 节点特征
            adj: (N, N) - 邻接矩阵
        
        Returns:
            out: (B*N, T, d_model) - 聚合后的特征
        """
        # x: (B*N, T, d_model) -> (B, N, T, d_model)
        B_N, T, C = x.shape
        N = adj.shape[0]
        B = B_N // N
        
        x_spatial = x.reshape(B, N, T, C)
        # 对空间维度进行图卷积: (B, N, T, C) @ (N, N) -> (B, N, T, C)
        out = torch.einsum('bntc,nm->bmtc', x_spatial, adj)
        out = self.dropout(out)
        return out.reshape(B_N, T, C)


class DynamicGraphLearning(nn.Module):
    """
    动态图学习模块 - 受DGCRN启发
    从节点嵌入动态生成图结构
    """
    def __init__(self, node_num, node_dim, alpha=0.1, tanhalpha=3.0):
        super().__init__()
        self.node_num = node_num
        self.node_dim = node_dim
        self.alpha = alpha
        self.tanhalpha = tanhalpha
        
        # 节点嵌入向量 (受DGCRN的emb1和emb2启发)
        self.emb1 = nn.Embedding(node_num, node_dim)
        self.emb2 = nn.Embedding(node_num, node_dim)
        
    def forward(self, h=None):
        """
        动态生成图结构
        Args:
            h: 隐藏状态 (可选，暂不使用)
        
        Returns:
            adj: (N, N) - 学到的邻接矩阵 (行和为1的概率矩阵)
        """
        idx = torch.arange(self.node_num, dtype=torch.long)
        if h is not None:
            idx = idx.to(h.device)
        
        nodevec1 = self.emb1(idx)  # (N, node_dim)
        nodevec2 = self.emb2(idx)  # (N, node_dim)
        
        # 计算节点间的关系强度
        a = torch.matmul(nodevec1, nodevec2.t()) - torch.matmul(nodevec2, nodevec1.t())
        adj = F.relu(torch.tanh(self.tanhalpha * a))
        
        # 归一化为行和为1的概率矩阵
        adj = adj + torch.eye(self.node_num, device=adj.device)
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        
        return adj


class SpatialAttention(nn.Module):
    """
    空间注意力机制 - 学习节点间的动态权重
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_flat, node_num):
        """
        Args:
            x_flat: (B*N, T, d_model)
            node_num: N
        
        Returns:
            out: (B*N, T, d_model)
        """
        B_N, T, C = x_flat.shape
        B = B_N // node_num
        
        # 重塑为 (B, N, T, d_model)
        x = x_flat.reshape(B, node_num, T, C)
        
        # 生成查询、键、值 (在节点维度上)
        Q = self.query(x)  # (B, N, T, d_model)
        K = self.key(x)
        V = self.value(x)
        
        # 多头注意力 (在节点维度上计算)
        # (B, N, T, num_heads, head_dim) -> (B, num_heads, N, T, head_dim)
        Q = Q.reshape(B, node_num, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = K.reshape(B, node_num, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.reshape(B, node_num, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # 计算注意力权重 (在节点维度)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-2)  # 在节点维度softmax
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, V)  # (B, num_heads, N, T, head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, node_num, T, C)
        out = out.reshape(B_N, T, C)
        
        out = self.fc_out(out)
        return out


class EncoderBlock(nn.Module):
    """
    改进的编码器块：融合Mamba序列建模和图卷积的空间建模。
    
    逐步降采样时间维度，同时通过Mamba层和GCN提取时空特征。
    输入形状: (B*N, T, d_model) - 批大小×节点数, 时间步长, 特征维度
    输出: 降采样特征 + 跳跃连接(用于解码器)
    """

    def __init__(self, d_model, n_mamba_layers=1, dropout=0.1, use_gcn=True, node_num=None):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.use_gcn = use_gcn
        self.node_num = node_num

        # Mamba层用于序列特征提取
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )
        
        # 添加图卷积层（如果启用）
        if self.use_gcn and node_num is not None:
            self.gcn = GCNLayer(d_model, dropout)
            self.spatial_attn = SpatialAttention(d_model, num_heads=4, dropout=dropout)

        # 时间降采样：使用步长=2的卷积
        self.downsample_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=2,
        )

        # 层归一化以稳定梯度流
        self.norm = nn.LayerNorm(d_model)
        self.norm_gcn = nn.LayerNorm(d_model) if self.use_gcn else None

        # Dropout用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        """
        Args:
            x: (B*N, T, d_model) - 输入特征序列
            adj: (N, N) - 邻接矩阵 (可选)

        Returns:
            x_down: (B*N, T//2, d_model) - 降采样后的特征
            x_skip: (B*N, T, d_model) - 原始特征用作跳跃连接
        """
        # 保存跳跃连接
        x_skip = x

        # 通过多个Mamba层进行时间序列建模
        for i, mamba in enumerate(self.mambas):
            x = mamba(x)  # (B*N, T, d_model)
            x = self.dropout(x)
        
        # 添加空间信息（通过GCN和注意力机制）
        if self.use_gcn and self.node_num is not None and adj is not None:
            # 应用图卷积
            x_gcn = self.gcn(x, adj)
            x_gcn = self.dropout(x_gcn)
            
            # 应用空间注意力
            x_attn = self.spatial_attn(x, self.node_num)
            x_attn = self.dropout(x_attn)
            
            # 融合时间和空间特征
            x = x + self.norm_gcn(x_gcn + x_attn)

        # 为卷积操作调整维度
        x = x.permute(0, 2, 1)

        # 进行时间降采样
        x = self.downsample_conv(x)

        # 恢复维度
        x = x.permute(0, 2, 1)

        # 应用层归一化
        x = self.norm(x)

        return x, x_skip


class DecoderBlock(nn.Module):
    """
    改进的解码器块：融合Mamba和图卷积的逐步上采样和特征融合。
    
    逐步上采样时间维度，并融合编码器的跳跃连接特征。
    输入: 降采样特征 + 来自编码器的跳跃连接
    输出: (B*N, T, d_model) - 恢复到原始时间长度的特征
    """

    def __init__(self, d_model, n_mamba_layers=1, dropout=0.1, use_gcn=True, node_num=None):
        super().__init__()
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.use_gcn = use_gcn
        self.node_num = node_num

        # 时间上采样：使用转置卷积
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 特征融合：在拼接后将通道数从2*d_model投影回d_model
        self.fusion_conv = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        # Mamba层进行上采样后的特征精化
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )
        
        # 添加图卷积层（如果启用）
        if self.use_gcn and node_num is not None:
            self.gcn = GCNLayer(d_model, dropout)
            self.spatial_attn = SpatialAttention(d_model, num_heads=4, dropout=dropout)

        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        self.norm_gcn = nn.LayerNorm(d_model) if self.use_gcn else None

        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip, adj=None):
        """
        Args:
            x: (B*N, T_low, d_model) - 来自编码器的降采样特征
            skip: (B*N, T_skip, d_model) - 来自编码器相同层级的跳跃连接
            adj: (N, N) - 邻接矩阵 (可选)

        Returns:
            out: (B*N, T_skip, d_model) - 融合后与skip同长度的特征
        """
        # 为转置卷积调整维度
        x = x.permute(0, 2, 1)

        # 时间上采样
        x = self.upsample_conv(x)

        # 恢复维度用于后续处理
        x = x.permute(0, 2, 1)

        # ========== 对齐上采样和跳跃连接的时间维度 ==========
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

        # ========== 特征融合：拼接并投影 ==========
        fused = torch.cat(
            [x.permute(0, 2, 1), skip.permute(0, 2, 1)],
            dim=1,
        )

        fused = self.fusion_conv(fused)
        fused = fused.permute(0, 2, 1)

        # 应用层归一化
        fused = self.norm(fused)

        # ========== 通过Mamba层进一步精化特征 ==========
        out = fused
        for mamba in self.mambas:
            out = mamba(out)
            out = self.dropout(out)
        
        # 添加空间信息（通过GCN和注意力机制）
        if self.use_gcn and self.node_num is not None and adj is not None:
            # 应用图卷积
            out_gcn = self.gcn(out, adj)
            out_gcn = self.dropout(out_gcn)
            
            # 应用空间注意力
            out_attn = self.spatial_attn(out, self.node_num)
            out_attn = self.dropout(out_attn)
            
            # 融合特征
            out = out + self.norm_gcn(out_gcn + out_attn)

        return out


class UMamba(BaseModel):
    """
    改进的U-Mamba模型：融合图学习的时空序列预测架构。
    
    核心架构:
    1. 动态图学习：从节点嵌入学习图结构
    2. 输入投影：将原始特征投影到高维Mamba空间
    3. 编码器：多层级逐步降采样，提取多尺度时空特征
    4. 瓶颈：在最低分辨率进行深度特征处理
    5. 解码器：多层级逐步上采样，融合跳跃连接
    6. 时间投影：将序列长度投影到预测长度
    7. 输出投影：恢复到原始特征维度
    
    输入: (B, T, N, F) - 批大小, 时间步长, 节点数, 特征数
    输出: (B, H, N, F) - 批大小, 预测长度, 节点数, 特征数
    """

    def __init__(
        self,
        d_model,
        feature,
        num_levels=3,
        n_mamba_per_block=1,
        dropout=0.1,
        use_graph_learning=True,
        node_dim=32,
        tanhalpha=3.0,
        **args
    ):
        """
        Args:
            d_model: Mamba和中间层的隐藏维度（推荐64-128）
            feature: 输入特征维度（通常为1）
            node_num: 节点数量
            num_levels: U-Net的层级数（推荐3-4）
            n_mamba_per_block: 每个编码/解码块中Mamba层的数量（推荐1-2）
            dropout: Dropout比率（推荐0.1-0.3）
            use_graph_learning: 是否使用动态图学习
            node_dim: 节点嵌入维度（推荐16-64）
            tanhalpha: 图学习中tanh的缩放因子
            **args: 其他参数（input_dim, output_dim, seq_len, horizon等）
        """
        super(UMamba, self).__init__(**args)
        self.d_model = d_model
        self.feature = feature
        self.num_levels = num_levels
        self.n_mamba_per_block = n_mamba_per_block
        self.dropout = dropout
        self.use_graph_learning = use_graph_learning
        self.node_dim = node_dim
        self.tanhalpha = tanhalpha

        # ========== 动态图学习模块 ==========
        if self.use_graph_learning:
            self.graph_learning = DynamicGraphLearning(
                node_num=self.node_num,
                node_dim=node_dim,
                tanhalpha=tanhalpha
            )

        # ========== 输入/输出投影 ==========
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.feature)

        # ========== 编码器堆栈 ==========
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    self.d_model,
                    n_mamba_layers=self.n_mamba_per_block,
                    dropout=self.dropout,
                    use_gcn=self.use_graph_learning,
                    node_num=self.node_num,
                )
                for _ in range(self.num_levels)
            ]
        )

        # ========== 解码器堆栈 ==========
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    self.d_model,
                    n_mamba_layers=self.n_mamba_per_block,
                    dropout=self.dropout,
                    use_gcn=self.use_graph_learning,
                    node_num=self.node_num,
                )
                for _ in range(self.num_levels)
            ]
        )

        # ========== 瓶颈层 ==========
        self.bottleneck = nn.ModuleList(
            [Mamba(d_model=self.d_model) for _ in range(self.n_mamba_per_block)]
        )
        self.bottleneck_gcn = GCNLayer(self.d_model, dropout) if self.use_graph_learning else None
        self.bottleneck_attn = SpatialAttention(self.d_model, num_heads=4, dropout=dropout) if self.use_graph_learning else None
        self.bottleneck_norm = nn.LayerNorm(self.d_model)
        self.bottleneck_norm_graph = nn.LayerNorm(self.d_model) if self.use_graph_learning else None
        self.bottleneck_dropout = nn.Dropout(dropout)

        # ========== 时间投影层 ==========
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

        # ========== 获取学习的图结构 ==========
        if self.use_graph_learning:
            adj = self.graph_learning(x)
        else:
            adj = None

        # ========== 1. 空间维度折叠 ==========
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x = x.reshape(B * N, T, F)  # (B*N, T, F)

        # ========== 2. 特征投影到高维空间 ==========
        x = self.input_proj(x)  # (B*N, T, d_model)

        # ========== 3. 编码器：多尺度特征提取 ==========
        skips = []
        cur = x

        for level, encoder in enumerate(self.encoders):
            if self.use_graph_learning and adj is not None:
                cur, skip = encoder(cur, adj=adj)
            else:
                cur, skip = encoder(cur, adj=None)
            skips.append(skip)

        # ========== 4. 瓶颈层：深度特征处理 ==========
        for mamba in self.bottleneck:
            cur = mamba(cur)

        cur = self.bottleneck_norm(cur)
        
        # 在瓶颈处添加图卷积
        if self.use_graph_learning and adj is not None and self.bottleneck_gcn is not None:
            cur_gcn = self.bottleneck_gcn(cur, adj)
            cur_attn = self.bottleneck_attn(cur, self.node_num)
            cur = cur + self.bottleneck_norm_graph(cur_gcn + cur_attn)
        
        cur = self.bottleneck_dropout(cur)

        # ========== 5. 解码器：逐步上采样和特征融合 ==========
        for decoder, skip in zip(self.decoders, reversed(skips)):
            if self.use_graph_learning and adj is not None:
                cur = decoder(cur, skip, adj=adj)
            else:
                cur = decoder(cur, skip, adj=None)

        # ========== 6. 对齐时间维度 ==========
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

        # ========== 7. 时间投影：从历史长度到预测长度 ==========
        cur = cur.permute(0, 2, 1)  # (B*N, d_model, T)
        cur = self.time_proj(cur)  # (B*N, d_model, H)
        cur = cur.permute(0, 2, 1)  # (B*N, H, d_model)

        # ========== 8. 特征投影回原始维度 ==========
        cur = self.output_proj(cur)  # (B*N, H, F)

        # ========== 9. 空间维度展开 ==========
        cur = cur.reshape(B, N, self.horizon, F)  # (B, N, H, F)
        cur = cur.permute(0, 2, 1, 3)  # (B, H, N, F)

        return cur
