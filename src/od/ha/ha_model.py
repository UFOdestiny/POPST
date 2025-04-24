import torch.nn as nn
import torch
from base.model import BaseModel


class HA(BaseModel):
    def __init__(self, **args):
        super(HA, self).__init__(**args)

    def forward(self, input, label=None):  # (b, t, n, f)
        x = input.permute(0, 2, 3, 1)
        x = x.mean(dim=-1, keepdim=True)
        x = x.permute(0, 3, 1, 2)
        return x
