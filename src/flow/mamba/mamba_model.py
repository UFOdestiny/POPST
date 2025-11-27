import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Residual Mamba block with lightweight normalization."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class myMamba(BaseModel):
    def __init__(self, d_model, num_layers, feature, depth=2, dropout=0.1, **args):
        super(myMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = max(1, num_layers)
        self.feature = feature
        self.depth = max(1, depth)
        self.dropout = dropout

        # project F → d_model
        self.input_proj = nn.Linear(self.feature, self.d_model)

        # U-Net style encoder/decoder built from residual Mamba blocks
        self.encoder_stages = nn.ModuleList(
            [self._build_stage() for _ in range(self.depth)]
        )

        down_blocks = max(0, self.depth - 1)
        self.downsamples = nn.ModuleList(
            [
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
                for _ in range(down_blocks)
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    self.d_model, self.d_model, kernel_size=4, stride=2, padding=1
                )
                for _ in range(down_blocks)
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [self._build_stage() for _ in range(down_blocks)]
        )
        self.skip_projs = nn.ModuleList(
            [nn.Linear(self.d_model * 2, self.d_model) for _ in range(down_blocks)]
        )

        self.bottleneck_stage = self._build_stage()

        # project sequence length T → H
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

        # project d_model → F
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def _build_stage(self):
        return nn.Sequential(
            *[MambaBlock(self.d_model, self.dropout) for _ in range(self.num_layers)]
        )

    def _apply_downsample(self, x, layer):
        x = x.permute(0, 2, 1)
        x = layer(x)
        return x.permute(0, 2, 1)

    def _apply_upsample(self, x, layer):
        x = x.permute(0, 2, 1)
        x = layer(x)
        return x.permute(0, 2, 1)

    def _match_length(self, tensor, target_len):
        current_len = tensor.size(1)
        if current_len == target_len:
            return tensor
        if current_len > target_len:
            return tensor[:, :target_len, :]
        pad = target_len - current_len
        return F.pad(tensor, (0, 0, 0, pad))

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # merge batch and nodes → treat each node independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # feature projection
        x = self.input_proj(x)  # (B*N, T, d_model)

        # encoder
        skips = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if idx < len(self.downsamples):
                skips.append(x)
                x = self._apply_downsample(x, self.downsamples[idx])

        # bottleneck
        x = self.bottleneck_stage(x)

        # decoder with skip connections
        for stage, upsample, proj in zip(
            reversed(self.decoder_stages),
            reversed(self.upsamples),
            reversed(self.skip_projs),
        ):
            x = self._apply_upsample(x, upsample)
            skip = skips.pop()
            target_len = skip.size(1)
            x = self._match_length(x, target_len)
            skip = self._match_length(skip, target_len)
            x = torch.cat([x, skip], dim=-1)
            x = proj(x)
            x = stage(x)

        # make sure sequence length matches the original input for downstream projection
        x = self._match_length(x, self.seq_len)

        # project T → H
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.time_proj(x)  # (B*N, d_model, H)
        x = x.permute(0, 2, 1)  # (B*N, H, d_model)

        # project feature back
        x = self.output_proj(x)  # (B*N, H, F)

        # reshape back to (B, H, N, F)
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x