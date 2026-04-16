import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


class LSTM(BaseModel):
    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, **args):
        super(LSTM, self).__init__(**args)
        self.start_linear = nn.Linear(self.input_dim, init_dim)

        self.lstm = nn.LSTM(
            input_size=init_dim,
            hidden_size=hid_dim,
            num_layers=layer,
            batch_first=True,
            dropout=dropout,
        )

        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, self.horizon * self.output_dim)

    def forward(self, input, label=None):  # (b, t, n, f)
        b, t, n, f = input.shape

        x = input.permute(0, 2, 1, 3).contiguous()  # (B, N, T, F)
        x = x.view(b * n, t, f)                      # (B*N, T, F)

        x = self.start_linear(x)                      # (B*N, T, init_dim)
        out, _ = self.lstm(x)
        x = out[:, -1, :]                             # (B*N, hid_dim)

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)                        # (B*N, horizon * output_dim)

        x = x.view(b, n, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)                     # (B, Horizon, N, output_dim)

        return x