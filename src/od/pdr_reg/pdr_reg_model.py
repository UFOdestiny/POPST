"""Point-regression variant of PDR."""

import torch
import torch.nn as nn

from src.od.pdr.pdr_model import PDR


class ODRegimeRegressionHead(nn.Module):
    """PDR's mixture-of-experts head with one point output per OD pair."""

    def __init__(self, context_dim, hidden_dim=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.base = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, context):
        base = self.base(context)
        gate = nn.functional.softmax(self.router(context), dim=-1)
        deltas = torch.stack(
            [expert(context) for expert in self.experts], dim=-2
        )
        return base + (gate.unsqueeze(-1) * deltas).sum(dim=-2)


class PDRReg(PDR):
    """PDR encoder trained as an ordinary point-regression OD model.

    The probabilistic PDR head emits three ZINB parameters.  This variant keeps
    the same encoder, router, and expert layout, but each base/expert branch
    emits one unconstrained regression value.  Consequently ``forward``
    returns the standard OD tensor expected by ``BaseEngine_OD`` and can be
    trained with ordinary losses such as MAE or MSE.
    """

    def __init__(
        self,
        *args,
        context_dim=64,
        num_experts=3,
        head_hidden_dim=128,
        dropout=0.0,
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
        self.head = ODRegimeRegressionHead(
            context_dim=context_dim,
            hidden_dim=head_hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

    def forward(self, X, label=None):
        """Return point forecasts shaped ``(B, horizon, N, N, D)``."""
        X, batch_size, channels, squeeze_back = self._fold_channels(X)
        x = X.permute(0, 2, 3, 1)

        context = self._encode(x)
        pred = self.head(context)[..., 0].permute(0, 3, 1, 2)
        return self._unfold_channels(pred, batch_size, channels, squeeze_back)
