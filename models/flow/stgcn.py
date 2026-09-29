from models.layers import TemporalConvLayer, STConvBlock
import torch
import torch.nn as nn
from models.base import BaseModel
import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx


class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, node_num, dropout=0.0):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], node_num)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1])
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel)

        self.tc1_ln = nn.LayerNorm([node_num, channels[0]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (b, c_in, t, n)
        x = self.tmp_conv1(x)  # (b, channels[0], t, n)
        # Convert to (b, t, n, channels[0]) for layer norm and FC layers
        x = x.permute(0, 2, 3, 1)  # (b, t, n, channels[0])
        x = self.tc1_ln(x)  # (b, t, n, channels[0])
        x = self.fc1(x)  # (b, t, n, channels[1])
        x = self.relu(x)
        x = self.dropout(x)  # official applies dropout between relu and fc2
        x = self.fc2(x)  # (b, t, n, end_channel)
        # Convert back to (b, end_channel, t, n)
        x = x.permute(0, 3, 1, 2)  # (b, end_channel, t, n)
        return x


class STGCN(BaseModel):
    """
    Reference code: https://github.com/hazdzz/STGCN
    """

    def __init__(self, gso, blocks, Kt, Ks, dropout, feature, horizon, **args):
        super(STGCN, self).__init__(horizon=horizon, **args)
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(Kt, Ks, self.node_num, blocks[l][-1], blocks[l + 1], gso, dropout)
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = self.seq_len - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], feature, self.node_num, dropout
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0])
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0])
            self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.horizon = horizon

    def forward(self, x, label=None):  # (b, t, n, f)
        origin_x = x
        step = x.shape[1]

        result = None

        for i in range(self.horizon):
            x = x.permute(0, 3, 1, 2)  # (b, f, t, n) - reshape to channel-first format
            x = self.st_blocks(x)  # Process through ST blocks: (b, f, t, n)
            if self.Ko > 1:
                x = self.output(x)  # (b, f, t, n)
            elif self.Ko == 0:
                x = self.fc1(x.permute(0, 2, 3, 1))  # (b, t, n, f) -> FC layer
                x = self.relu(x)
                x = self.fc2(x).permute(0, 3, 1, 2)  # (b, f, t, n)

            # Take only the last timestep and convert back to (b, t, n, c) format
            # x is (b, c, t, n), take last timestep -> (b, c, 1, n); c = output_dim
            x_last = x[:, :, -1:, :]  # (b, c, 1, n)
            x_last = x_last.permute(0, 2, 3, 1)  # (b, 1, n, c)

            if result is None:
                result = x_last
            else:
                result = torch.cat([result, x_last], dim=1)  # concatenate along time dimension

            # Autoregressive feedback must keep the input's channel count
            # (input_dim).  Under CQR the output has 3 channels per feature
            # (q_lo, q_mid, q_hi); feed back only the median.  Reshaping the
            # last axis to (input_dim, -1) and taking [..., 0] yields the
            # median under CQR and is a no-op when output_dim == input_dim.
            x_feedback = x_last.reshape(*x_last.shape[:-1], origin_x.shape[-1], -1)[..., 0]

            origin_x = torch.cat([origin_x, x_feedback], dim=1)  # (b, t+1, n, input_dim)
            x = origin_x[:, -step:, :, :]  # get last 'step' timesteps

        return result


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "scalap")[0]
    gso = torch.tensor(gso).to(device)
    return {"gso": gso}


def build_model(config, node_num, **ctx):
    Ko = config.data.seq_len - (config.model.params.Kt - 1) * 2 * config.model.params.block_num
    blocks = []
    blocks.append([config.data.input_dim])
    for l in range(config.model.params.block_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([config.data.output_dim])
    return STGCN(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        gso=ctx["gso"],
        blocks=blocks,
        Kt=config.model.params.Kt,
        Ks=config.model.params.Ks,
        dropout=config.model.params.dropout,
        feature=config.data.output_dim,
        horizon=config.data.horizon,
        seq_len=config.data.seq_len,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, setup=setup)
