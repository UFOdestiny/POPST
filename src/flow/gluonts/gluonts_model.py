import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


class TemporalBlock(nn.Module):
    """Temporal convolution block for extracting temporal features."""
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
        
        # residual connection
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
    """Temporal convolutional network, similar to the TCN structure in GluonTS."""
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
    Time series forecasting model based on the GluonTS design philosophy.

    Combines the following features:
    1. Temporal convolutional network (TCN) to capture long-term dependencies
    2. Self-attention mechanism to model temporal relationships
    3. Probabilistic prediction head for uncertainty estimation
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
        
        # input projection layer
        self.input_projection = nn.Linear(self.input_dim, init_dim)
        
        # temporal convolutional network
        hidden_dims = [hid_dim] * num_layers
        self.tcn = TemporalConvNet(
            input_dim=init_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # self-attention layer (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hid_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hid_dim)
        
        # output layer
        self.output_fc1 = nn.Linear(hid_dim, end_dim)
        self.output_fc2 = nn.Linear(end_dim, self.output_dim * self.horizon)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, label=None):
        """
        Forward pass.

        Args:
            x: input tensor of shape (batch, seq_len, node_num, input_dim)
            label: label (optional)

        Returns:
            output: predictions of shape (batch, horizon, node_num, output_dim)
        """
        batch_size, seq_len, node_num, input_dim = x.shape

        # reshape input: (batch * node_num, seq_len, input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, node_num, seq_len, input_dim)
        x = x.view(batch_size * node_num, seq_len, input_dim)

        # input projection: (batch * node_num, seq_len, init_dim)
        x = self.input_projection(x)

        # TCN expects input format (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch * node_num, init_dim, seq_len)
        x = self.tcn(x)  # (batch * node_num, hid_dim, seq_len)

        # convert back to (batch * node_num, seq_len, hid_dim)
        x = x.permute(0, 2, 1)

        # self-attention (optional)
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)

        # take the features of the last time step
        x = x[:, -1, :]  # (batch * node_num, hid_dim)

        # output mapping
        x = F.relu(self.output_fc1(x))
        x = self.dropout(x)
        x = self.output_fc2(x)  # (batch * node_num, output_dim * horizon)

        # reshape output: (batch, horizon, node_num, output_dim)
        x = x.view(batch_size, node_num, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (batch, horizon, node_num, output_dim)
        
        return x


class ProbabilisticGluonTSModel(BaseModel):
    """
    Probabilistic forecasting version of the GluonTS model.

    Outputs mean and standard deviation, supporting probabilistic prediction.
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
        
        # input projection layer
        self.input_projection = nn.Linear(self.input_dim, init_dim)
        
        # temporal convolutional network
        hidden_dims = [hid_dim] * num_layers
        self.tcn = TemporalConvNet(
            input_dim=init_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # self-attention layer (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hid_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hid_dim)
        
        # output layer - mean
        self.mean_fc1 = nn.Linear(hid_dim, end_dim)
        self.mean_fc2 = nn.Linear(end_dim, self.output_dim * self.horizon)
        
        # output layer - standard deviation
        self.std_fc1 = nn.Linear(hid_dim, end_dim)
        self.std_fc2 = nn.Linear(end_dim, self.output_dim * self.horizon)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, label=None):
        """
        Forward pass.

        Args:
            x: input tensor of shape (batch, seq_len, node_num, input_dim)
            label: label (optional)

        Returns:
            mean: predicted mean of shape (batch, horizon, node_num, output_dim)
            std: predicted standard deviation of shape (batch, horizon, node_num, output_dim)
        """
        batch_size, seq_len, node_num, input_dim = x.shape

        # reshape input
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * node_num, seq_len, input_dim)

        # input projection
        x = self.input_projection(x)

        # TCN
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)

        # self-attention
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)

        # take the last time step
        x = x[:, -1, :]

        # mean output
        mean = F.relu(self.mean_fc1(x))
        mean = self.dropout(mean)
        mean = self.mean_fc2(mean)

        # standard deviation output (ensure positivity)
        std = F.relu(self.std_fc1(x))
        std = self.dropout(std)
        std = F.softplus(self.std_fc2(std))  # ensure standard deviation is positive

        # reshape output
        mean = mean.view(batch_size, node_num, self.horizon, self.output_dim)
        mean = mean.permute(0, 2, 1, 3)
        
        std = std.view(batch_size, node_num, self.horizon, self.output_dim)
        std = std.permute(0, 2, 1, 3)
        
        if self.training:
            return mean, std
        else:
            return mean
