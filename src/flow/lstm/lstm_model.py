import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


class LSTM(BaseModel):
    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, **args):
        super(LSTM, self).__init__(**args)
        # 不需要 start_conv，因为 LSTM 直接吃 (Seq, Input_Dim)
        # 如果你想把特征先映射一下，可以用 Linear
        self.start_linear = nn.Linear(args['input_dim'], init_dim) # 假设 input_dim 是 F

        self.lstm = nn.LSTM(
            input_size=init_dim,
            hidden_size=hid_dim,
            num_layers=layer,
            batch_first=True,
            dropout=dropout
        )

        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, self.horizon)

    def forward(self, input, label=None):
        # input shape: (B, T, N, F)
        b, t, n, f = input.shape
        
        # 1. 调整维度以适应 LSTM
        # 我们需要把 (B, N) 合并，作为 batch 处理，T 作为序列长度
        # 目标: (B*N, T, F)
        x = input.permute(0, 2, 1, 3).contiguous()  # (B, N, T, F)
        x = x.view(b * n, t, f)                     # (B*N, T, F)

        # 2. 特征映射 (可选)
        x = self.start_linear(x)                    # (B*N, T, init_dim)
        
        # 3. LSTM
        # out: (B*N, T, hid_dim), _ : (h_n, c_n)
        out, _ = self.lstm(x)
        
        # 4. 取最后一个时间步的输出用于预测未来
        # 或者如果你是 Seq2Seq，可能处理方式不同。这里假设用最后一步预测未来
        x = out[:, -1, :]                           # (B*N, hid_dim)

        # 5. 输出层
        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)                     # (B*N, horizon)

        # 6. 恢复维度 (B, N, Horizon) 或 (B, Horizon, N)
        x = x.view(b, n, self.horizon)
        x = x.permute(0, 2, 1).unsqueeze(-1)        # 变成 (B, Horizon, N, 1) 或者你需要的格式
        
        return x