import torch
import torch.nn as nn
from base.model import BaseModel


class myTransformer(BaseModel):
    """
    Transformer baseline
    Input:  (B, T, N, F)
    Output: (B, H, N, F)
    """

    def __init__(self, num_layers, d_model, n_heads, dropout, feature, **args):
        super(myTransformer, self).__init__(**args)
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.feature = feature
        self.d_model = d_model

        # **Project input features → d_model**
        # (F → d_model)
        self.input_proj = nn.Linear(self.feature, self.d_model)

        # Positional encoding for time
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, 1, self.d_model))

        # Transformer encoder (temporal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # **Project back to original feature dim**
        # (d_model → F)
        self.output_proj = nn.Linear(self.d_model, self.feature)

        # **Sequence length projection (T → H)**
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

    def forward(self, x):  # x: (B, T, N, F)
        B, T, N, F = x.shape

        # Merge batch + nodes → treat each node independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # Feature projection
        x = self.input_proj(x)  # (B*N, T, d_model)

        # Add temporal position embedding
        x = x + self.pos_embed[:, :T, :, :].reshape(1, T, self.d_model)

        # Transformer encoding
        x = self.transformer(x)  # (B*N, T, d_model)

        # === Temporal projection: T → H ===
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.time_proj(x)  # (B*N, d_model, H)
        x = x.permute(0, 2, 1)  # (B*N, H, d_model)

        # === Restore feature dim ===
        x = self.output_proj(x)  # (B*N, H, F)

        # === Reshape back to (B, H, N, F) ===
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)
        return x
