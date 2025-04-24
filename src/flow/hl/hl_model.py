import torch.nn as nn
from base.model import BaseModel

class HL(BaseModel):
    def __init__(self, **args):
        super(HL, self).__init__(**args)
        self.L = nn.Linear(self.seq_len, self.horizon)

    def forward(self, input, label=None):  # (b, t, n, f)
        x = input.permute(0, 2, 3, 1)
        x = self.L(x)
        x = x.permute(0, 3, 1, 2)
        return x