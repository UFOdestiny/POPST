"""Shared probabilistic-regression model for the PDR distribution baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.od.pdr.pdr_model import PDR


class ODRegimeDistributionHead(nn.Module):
    """Mixture-of-experts head emitting a location and a raw scale."""

    def __init__(self, context_dim, hidden_dim=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.base = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.router = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, context):
        base = self.base(context)
        gate = F.softmax(self.router(context), dim=-1)
        deltas = torch.stack(
            [expert(context) for expert in self.experts], dim=-2
        )
        return base + (gate.unsqueeze(-1) * deltas).sum(dim=-2)


class PDRRegDistribution(PDR):
    """PDR encoder with a heteroscedastic location-scale regression head.

    Gaussian, Laplace, and Student-t baselines share this exact model.  Their
    only difference is the likelihood used by the engine, making the ablation
    a controlled comparison of predictive distributions.
    """

    cqr_compatible = False

    def __init__(
        self,
        *args,
        context_dim=64,
        num_experts=3,
        head_hidden_dim=128,
        dropout=0.0,
        min_scale=1e-4,
        **kwargs,
    ):
        super().__init__(
            *args,
            context_dim=context_dim,
            num_experts=num_experts,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            **kwargs,
        )
        self.min_scale = float(min_scale)
        if self.min_scale <= 0.0:
            raise ValueError("min_scale must be positive")
        self.head = ODRegimeDistributionHead(
            context_dim=context_dim,
            hidden_dim=head_hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

    def forward(self, X, label=None):
        """Return ``(location, scale)`` in ``(B, horizon, N, N, D)`` layout."""
        X, batch_size, channels, squeeze_back = self._fold_channels(X)
        x = X.permute(0, 2, 3, 1)

        raw = self.head(self._encode(x))
        loc = raw[..., 0].permute(0, 3, 1, 2)
        scale = (F.softplus(raw[..., 1]) + self.min_scale).permute(0, 3, 1, 2)

        loc = self._unfold_channels(loc, batch_size, channels, squeeze_back)
        scale = self._unfold_channels(scale, batch_size, channels, squeeze_back)
        return loc, scale
