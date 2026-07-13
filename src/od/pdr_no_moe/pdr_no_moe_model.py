"""PDR ablation replacing the mixture of experts with one MLP head."""

import torch.nn as nn

from src.od.pdr.pdr_model import PDR


class ODSingleHead(nn.Module):
    """Single ZINB-parameter head matched to PDR's shared base head."""

    def __init__(self, context_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, context):
        return self.net(context)


class PDR_no_moe(PDR):
    """Retain the PDR encoder but remove its router and expert residuals."""

    def __init__(
        self,
        *args,
        context_dim=64,
        head_hidden_dim=128,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            context_dim=context_dim,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            **kwargs,
        )
        self.head = ODSingleHead(context_dim, head_hidden_dim, dropout)
