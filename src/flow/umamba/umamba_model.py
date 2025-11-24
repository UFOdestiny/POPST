import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_mamba_layers=1):
        super().__init__()
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )
        # downsample conv on time: in_channels=d_model, out_channels=d_model, stride=2
        self.down = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=2,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B*N, T, d_model)
        for m in self.mambas:
            x = m(x)  # expect (B*N, T, d_model)
        # conv expects (B, C, L)
        x_c = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x_down = self.down(x_c)  # (B*N, d_model, T_down)
        x_down = x_down.permute(0, 2, 1)  # (B*N, T_down, d_model)
        x_down = self.norm(x_down)
        return (
            x_down,
            x,
        )  # return downsampled and skip (pre-downsample) for skip-connection


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_mamba_layers=1):
        super().__init__()
        # upsample convtranspose: doubles time dimension
        self.up = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        # after concat along channel (2*d_model) -> project back to d_model via 1x1 conv
        self.merge_conv = nn.Conv1d(
            in_channels=2 * d_model, out_channels=d_model, kernel_size=1
        )
        self.mambas = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_mamba_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, skip):
        # x: (B*N, T_low, d_model), skip: (B*N, T_skip, d_model)
        x_c = x.permute(0, 2, 1)  # (B*N, d_model, T_low)
        x_up = self.up(x_c)  # (B*N, d_model, T_up) -- ideally equals T_skip
        x_up = x_up.permute(0, 2, 1)  # (B*N, T_up, d_model)

        # if lengths mismatch, align by interpolation or cropping
        T_up = x_up.shape[1]
        T_skip = skip.shape[1]
        if T_up != T_skip:
            # try to align: if T_up < T_skip -> interpolate x_up; if > -> crop
            if T_up < T_skip:
                x_up = F.interpolate(
                    x_up.permute(0, 2, 1),
                    size=T_skip,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            else:
                x_up = x_up[:, :T_skip, :]

        # concat on channel axis: conv expects (B, C, L)
        cat = torch.cat(
            [x_up.permute(0, 2, 1), skip.permute(0, 2, 1)], dim=1
        )  # (B*N, 2*d_model, T)
        merged = self.merge_conv(cat)  # (B*N, d_model, T)
        merged = merged.permute(0, 2, 1)  # (B*N, T, d_model)
        merged = self.norm(merged)

        out = merged
        for m in self.mambas:
            out = m(out)  # (B*N, T, d_model)
        return out


class UMamba(BaseModel):
    def __init__(
        self, d_model, feature, num_levels=3, n_mamba_per_block=1, **args
    ):
        super(UMamba, self).__init__(**args)
        self.d_model = d_model
        self.feature = feature
        self.num_levels = num_levels
        self.n_mamba_per_block = n_mamba_per_block

        # projections
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.feature)

        # build encoder and decoder stacks
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(self.d_model, n_mamba_layers=self.n_mamba_per_block)
                for _ in range(self.num_levels)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(self.d_model, n_mamba_layers=self.n_mamba_per_block)
                for _ in range(self.num_levels)
            ]
        )

        # bottleneck Mamba layers (after lowest downsample)
        self.bottleneck = nn.ModuleList(
            [Mamba(d_model=self.d_model) for _ in range(self.n_mamba_per_block)]
        )
        self.bottleneck_norm = nn.LayerNorm(self.d_model)

        # temporal projection from seq_len -> horizon
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

    def forward(self, x):  # x: (B, T, N, F)
        B, T, N, F = x.shape
        # merge batch and nodes
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # input feature projection
        x = self.input_proj(x)  # (B*N, T, d_model)

        # ---- Encoder ----
        skips = []
        cur = x
        for enc in self.encoders:
            cur, skip = enc(cur)  # cur is downsampled, skip is pre-downsample
            skips.append(skip)

        # ---- Bottleneck ----
        for m in self.bottleneck:
            cur = m(cur)
        cur = self.bottleneck_norm(cur)

        # ---- Decoder (reverse order) ----
        for dec, skip in zip(self.decoders, reversed(skips)):
            cur = dec(cur, skip)

        # Now cur should be (B*N, T_recon, d_model). Ideally T_recon == original T
        T_recon = cur.shape[1]
        if T_recon != T:
            # try to align to original T by interpolation or cropping
            if T_recon < T:
                cur = F.interpolate(
                    cur.permute(0, 2, 1), size=T, mode="linear", align_corners=False
                ).permute(0, 2, 1)
            else:
                cur = cur[:, :T, :]

        # ---- time projection T -> H ----
        cur = cur.permute(0, 2, 1)  # (B*N, d_model, T)
        cur = self.time_proj(cur)  # (B*N, d_model, H)
        cur = cur.permute(0, 2, 1)  # (B*N, H, d_model)

        # project back to original feature dim
        cur = self.output_proj(cur)  # (B*N, H, F)

        # reshape back to (B, H, N, F)
        cur = cur.reshape(B, N, self.horizon, F).permute(0, 2, 1, 3)
        return cur
