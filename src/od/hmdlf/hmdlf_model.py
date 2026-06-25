import torch
import torch.nn as nn
from base.model import BaseODModel


class FlowEncoder(nn.Module):
    def __init__(self, seq_len, cnn_out, gru_hidden):
        super(FlowEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_out, cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(cnn_out * seq_len, gru_hidden)
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_proj = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x):  # x: (B*M*N, T)
        x = x.unsqueeze(1)  # (B*M*N, 1, T)
        x = self.cnn(x)  # (B*M*N, cnn_out, T)
        x = x.flatten(start_dim=1)  # (B*M*N, cnn_out * T)
        x = self.fc(x).unsqueeze(1)  # (B*M*N, 1, gru_hidden)
        x = x.repeat(1, 1, 1)  # (B*M*N, 1, gru_hidden)
        out, _ = self.gru(x)  # (B*M*N, 1, 2*gru_hidden)
        attn = torch.softmax(self.attn_proj(out), dim=1)  # (B*M*N, 1, 1)
        rep = (attn * out).sum(dim=1)  # (B*M*N, 2*gru_hidden)
        return rep


class HMDLF(BaseODModel):
    def __init__(
        self,
        node_num,
        input_dim,
        output_dim,
        seq_len,
        horizon,
        cnn_out=8,
        gru_hidden=16,
        predictor_hidden=32,
        num_mobility=1,
        share_encoder=True,
    ):
        super(HMDLF, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)
        self.share_encoder = share_encoder
        self.num_mobility = num_mobility
        self.output_dim = output_dim

        if share_encoder:
            self.encoder = FlowEncoder(seq_len, cnn_out, gru_hidden)
        else:
            self.encoders = nn.ModuleList(
                [FlowEncoder(seq_len, cnn_out, gru_hidden) for _ in range(num_mobility)]
            )

        # Per OD-cell forecast: predict ``horizon`` future scalars from the
        # cell's own history (not node_num).
        self.predictor = nn.Sequential(
            nn.Linear(gru_hidden * 2, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, horizon),
        )

    def forward_single(self, x, label=None):  # x: (B', T, M, N), M == N
        B, T, M, N = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B', M, N, T)

        if self.share_encoder:
            x = x.view(B * M * N, T)
            rep = self.encoder(x)
        else:
            x = x.view(B, M, N, T)
            reps = []
            for i in range(M):
                xi = x[:, i].reshape(B * N, T)  # (B'*N, T)
                rep_i = self.encoders[i](xi)  # (B'*N, 2*hidden)
                reps.append(rep_i)
            rep = torch.cat(reps, dim=0)  # (B'*M*N, 2*hidden)

        out = self.predictor(rep)  # (B'*M*N, horizon)
        out = out.view(B, M, N, self.horizon).permute(
            0, 3, 1, 2
        )  # (B', horizon, M, N)
        return out