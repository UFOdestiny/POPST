import torch
import torch.nn as nn
from base.model import BaseModel
from mamba_ssm import Mamba


class myMamba(BaseModel):
    def __init__(self, d_model, num_layers, feature, **args):
        super(myMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers

        self.seq_len = self.seq_len
        self.horizon = self.horizon
        self.feature = feature

        # project F → d_model
        self.input_proj = nn.Linear(self.feature, self.d_model)

        # stack Mamba layers for temporal modeling
        self.mamba_layers = nn.ModuleList(
            [Mamba(d_model=self.d_model) for _ in range(self.num_layers)]
        )

        # project sequence length T → H
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

        # project d_model → F
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # merge batch and nodes → treat each node independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # feature projection
        x = self.input_proj(x)  # (B*N, T, d_model)

        # Mamba temporal modeling
        for layer in self.mamba_layers:
            x = layer(x)  # still (B*N, T, d_model)

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