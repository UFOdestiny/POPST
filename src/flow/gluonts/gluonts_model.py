import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


class TemporalBlock(nn.Module):
    """时序卷积块，用于提取时序特征"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = self.relu(out + residual)
        return out


class TemporalConvNet(nn.Module):
    """时序卷积网络，类似于GluonTS中的TCN结构"""
    def __init__(self, input_dim, hidden_dims, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(hidden_dims)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GluonTSModel(BaseModel):
    """
    基于GluonTS设计理念的时间序列预测模型
    
    结合了以下特性：
    1. 时序卷积网络 (TCN) 用于捕获长期依赖
    2. 自注意力机制用于建模时序关系
    3. 概率预测头用于不确定性估计
    """
    def __init__(
        self,
        init_dim=32,
        hid_dim=64,
        end_dim=128,
        num_layers=3,
        kernel_size=3,
        dropout=0.1,
        num_heads=4,
        use_attention=True,
        **args
    ):
        super(GluonTSModel, self).__init__(**args)
        
        self.use_attention = use_attention
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, init_dim)
        
        # 时序卷积网络
        hidden_dims = [hid_dim] * num_layers
        self.tcn = TemporalConvNet(
            input_dim=init_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 自注意力层（可选）
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hid_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hid_dim)
        
        # 输出层
        self.output_fc1 = nn.Linear(hid_dim, end_dim)
        self.output_fc2 = nn.Linear(end_dim, self.output_dim * self.horizon)
        
        self.dropout = nn.Dropout(dropout)
        
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
        
        # 输入投影: (batch * node_num, seq_len, init_dim)
        x = self.input_projection(x)
        
        # TCN期望输入格式为 (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch * node_num, init_dim, seq_len)
        x = self.tcn(x)  # (batch * node_num, hid_dim, seq_len)
        
        # 转换回 (batch * node_num, seq_len, hid_dim)
        x = x.permute(0, 2, 1)
        
        # 自注意力（可选）
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)
        
        # 取最后一个时间步的特征
        x = x[:, -1, :]  # (batch * node_num, hid_dim)
        
        # 输出映射
        x = F.relu(self.output_fc1(x))
        x = self.dropout(x)
        x = self.output_fc2(x)  # (batch * node_num, output_dim * horizon)
        
        # 重塑输出: (batch, horizon, node_num, output_dim)
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (batch, horizon, node_num, output_dim)
        
        return x


class ProbabilisticGluonTSModel(BaseModel):
    """
    概率预测版本的GluonTS模型
    
    输出均值和标准差，支持概率预测
    """
    def __init__(
        self,
        init_dim=32,
        hid_dim=64,
        end_dim=128,
        num_layers=3,
        kernel_size=3,
        dropout=0.1,
        num_heads=4,
        use_attention=True,
        **args
    ):
        super(ProbabilisticGluonTSModel, self).__init__(**args)
        
        self.use_attention = use_attention
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, init_dim)
        
        # 时序卷积网络
        hidden_dims = [hid_dim] * num_layers
        self.tcn = TemporalConvNet(
            input_dim=init_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 自注意力层（可选）
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hid_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hid_dim)
        
        # 输出层 - 均值
        self.mean_fc1 = nn.Linear(hid_dim, end_dim)
        self.mean_fc2 = nn.Linear(end_dim, self.output_dim * self.horizon)
        
        # 输出层 - 标准差
        self.std_fc1 = nn.Linear(hid_dim, end_dim)
        self.std_fc2 = nn.Linear(end_dim, self.output_dim * self.horizon)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, label=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, node_num, input_dim)
            label: 标签（可选）
            
        Returns:
            mean: 预测均值，形状为 (batch, horizon, node_num, output_dim)
            std: 预测标准差，形状为 (batch, horizon, node_num, output_dim)
        """
        batch_size, seq_len, node_num, input_dim = x.shape
        
        # 重塑输入
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * node_num, seq_len, input_dim)
        
        # 输入投影
        x = self.input_projection(x)
        
        # TCN
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        
        # 自注意力
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # 均值输出
        mean = F.relu(self.mean_fc1(x))
        mean = self.dropout(mean)
        mean = self.mean_fc2(mean)
        
        # 标准差输出（确保为正）
        std = F.relu(self.std_fc1(x))
        std = self.dropout(std)
        std = F.softplus(self.std_fc2(std))  # 确保标准差为正
        
        # 重塑输出
        mean = mean.view(batch_size, node_num, self.horizon, self.output_dim)
        mean = mean.permute(0, 2, 1, 3)
        
        std = std.view(batch_size, node_num, self.horizon, self.output_dim)
        std = std.permute(0, 2, 1, 3)
        
        if self.training:
            return mean, std
        else:
            return mean
