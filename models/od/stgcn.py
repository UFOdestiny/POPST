from models.layers import TemporalConvLayer, STConvBlock
import torch
import torch.nn as nn
from models.base import BaseODModel
import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx


class OutputBlock(nn.Module):
    def __init__(self, Ko, Kt, last_block_channel, channels, end_channel, node_num):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], node_num)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1])
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel)

        self.tc1_ln = nn.LayerNorm([node_num, channels[0]])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = x.permute(0, 1, 3, 2)
        return x


class STGCN_OD(BaseODModel):
    """
    Reference code: https://github.com/hazdzz/STGCN
    """

    def __init__(self, gso, blocks, Kt, Ks, dropout, feature, horizon, **args):
        super(STGCN_OD, self).__init__(horizon=horizon, **args)
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(Kt, Ks, self.node_num, blocks[l][-1], blocks[l + 1], gso, dropout)
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = self.seq_len - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko >= 1:
            self.output = OutputBlock(Ko, Kt, blocks[-3][-1], blocks[-2], feature, self.node_num)
        else:
            raise ValueError("STGCN history is too short for the temporal blocks; increase seq_len")
        self.horizon = horizon

    def forward_single(self, x, label=None):  # (b, t, n, f)
        origin_x = x
        step = x.shape[1]

        result = None

        for i in range(self.horizon):
            x = x.permute(0, 3, 1, 2)  # b,f,t,n
            x = self.st_blocks(x)

            x = self.output(x)
            x = x.transpose(2, 3)

            if result is None:
                result = x
            else:
                result = torch.cat([result, x], dim=1)

            origin_x = torch.cat([origin_x, x], dim=1)
            x = origin_x[:, -step:, :, :]

        return result


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "scalap")[0]
    gso = torch.tensor(gso).to(device)
    Ko = config.data.seq_len - (config.model.params.Kt - 1) * 2 * config.model.params.block_num
    blocks = []
    blocks.append([config.data.input_dim])
    for l in range(config.model.params.block_num):
        blocks.append([64, 16, 64])
    if Ko < 1:
        raise ValueError("STGCN history is too short for the temporal blocks; increase seq_len")
    blocks.append([128, 128])
    blocks.append([config.data.input_dim])
    return dict(gso=gso, blocks=blocks)


def build_model(config, node_num, **ctx):
    return STGCN_OD(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        gso=ctx["gso"],
        blocks=ctx["blocks"],
        Kt=config.model.params.Kt,
        Ks=config.model.params.Ks,
        dropout=config.model.params.dropout,
        feature=config.data.input_dim,
        horizon=config.data.horizon,
        seq_len=config.data.seq_len,
    )


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        setup=setup,
        od=True,
        od_cqr=True,
    )
