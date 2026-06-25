import torch.nn as nn
from base.model import BaseODModel


class HL(BaseODModel):
    """Historical Linear: a learnable linear map over the input window, applied
    per origin-destination pair.  Channel-as-batch (see BaseODModel)."""

    def __init__(self, **args):
        super(HL, self).__init__(**args)
        self.L = nn.Linear(self.seq_len, self.horizon)

    def forward_single(self, x, label=None):  # (B', T, N, N)
        x = x.permute(0, 2, 3, 1)  # (B', N, N, T)
        x = self.L(x)              # (B', N, N, H)
        x = x.permute(0, 3, 1, 2)  # (B', H, N, N)
        return x
