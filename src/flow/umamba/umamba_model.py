import torch
import torch.nn as nn
from base.model import BaseModel
from mamba_ssm import Mamba


class MambaEncoderBlock(nn.Module):
    """Mamba encoder block with downsampling"""

    def __init__(self, d_model, downsample_factor=2, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.downsample_factor = downsample_factor
        # Downsample in time dimension
        self.downsample = nn.Linear(downsample_factor, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B, T, D)
        # Mamba processing with residual
        residual = x
        x = self.mamba(x)
        x = self.norm(x + residual)

        B, T, D = x.shape
        # Downsample: group consecutive time steps
        if T % self.downsample_factor != 0:
            # Pad if necessary
            pad_len = self.downsample_factor - (T % self.downsample_factor)
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
            T = T + pad_len

        x = x.reshape(B, T // self.downsample_factor, self.downsample_factor, D)
        x = x.permute(0, 1, 3, 2)  # (B, T//2, D, 2)
        x = self.downsample(x).squeeze(-1)  # (B, T//2, D)

        return x


class MambaDecoderBlock(nn.Module):
    """Mamba decoder block with upsampling and skip connection"""

    def __init__(self, d_model, upsample_factor=2, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.upsample_factor = upsample_factor
        # Upsample in time dimension
        self.upsample = nn.Linear(1, upsample_factor)
        # Fusion layer for skip connection
        self.skip_fusion = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):  # x: (B, T, D), skip: (B, T*2, D)
        B, T, D = x.shape

        # Upsample
        x = x.unsqueeze(-1)  # (B, T, D, 1)
        x = self.upsample(x)  # (B, T, D, 2)
        x = x.permute(0, 1, 3, 2)  # (B, T, 2, D)
        x = x.reshape(B, T * self.upsample_factor, D)  # (B, T*2, D)

        # Match skip connection size if needed
        if x.shape[1] != skip.shape[1]:
            # Interpolate to match
            x = x.permute(0, 2, 1)  # (B, D, T*2)
            x = nn.functional.interpolate(
                x, size=skip.shape[1], mode="linear", align_corners=False
            )
            x = x.permute(0, 2, 1)  # (B, T_skip, D)

        # Skip connection fusion
        x = torch.cat([x, skip], dim=-1)  # (B, T, D*2)
        x = self.skip_fusion(x)  # (B, T, D)

        # Mamba processing with residual
        residual = x
        x = self.mamba(x)
        x = self.norm(x + residual)

        return x


class MambaBottleneck(nn.Module):
    """Bottleneck Mamba block"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm(x + residual)
        return x


class UNetMamba(BaseModel):
    """
    UNet-style Mamba architecture for spatio-temporal prediction.

    Architecture:
    - Encoder: Multiple Mamba blocks with downsampling
    - Bottleneck: Mamba block at lowest resolution
    - Decoder: Multiple Mamba blocks with upsampling and skip connections
    """

    def __init__(
        self, d_model, num_layers, sample_factor, feature, dropout=0.1, **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers  # Number of encoder/decoder levels
        self.feature = feature

        # Input projection: F → d_model
        self.input_proj = nn.Linear(self.feature, self.d_model)

        # Encoder blocks (downsampling path)
        self.encoders = nn.ModuleList(
            [
                MambaEncoderBlock(
                    d_model=self.d_model,
                    downsample_factor=sample_factor,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Bottleneck
        self.bottleneck = MambaBottleneck(d_model=self.d_model, dropout=dropout)

        # Decoder blocks (upsampling path with skip connections)
        self.decoders = nn.ModuleList(
            [
                MambaDecoderBlock(
                    d_model=self.d_model, upsample_factor=sample_factor, dropout=dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output projection layers
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

        # Final refinement Mamba layer
        self.final_mamba = Mamba(d_model=self.d_model)
        self.final_norm = nn.LayerNorm(self.d_model)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # Merge batch and nodes → treat each node independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # Input projection
        x = self.input_proj(x)  # (B*N, T, d_model)

        # Store original for global skip
        x_input = x

        # ========== Encoder Path ==========
        skip_connections = []
        for encoder in self.encoders:
            skip_connections.append(x)  # Store before downsampling
            x = encoder(x)

        # ========== Bottleneck ==========
        x = self.bottleneck(x)

        # ========== Decoder Path ==========
        for i, decoder in enumerate(self.decoders):
            # Get corresponding skip connection (reverse order)
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)

        # Match original sequence length if needed
        if x.shape[1] != T:
            x = x.permute(0, 2, 1)  # (B*N, d_model, T')
            x = nn.functional.interpolate(x, size=T, mode="linear", align_corners=False)
            x = x.permute(0, 2, 1)  # (B*N, T, d_model)

        # Global skip connection
        x = x + x_input

        # Final refinement
        x = self.final_mamba(x)
        x = self.final_norm(x)

        # Project T → H
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.time_proj(x)  # (B*N, d_model, H)
        x = x.permute(0, 2, 1)  # (B*N, H, d_model)

        # Project feature back
        x = self.output_proj(x)  # (B*N, H, F)

        # Reshape back to (B, H, N, F)
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
