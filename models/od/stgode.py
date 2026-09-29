from models.layers import STGCNBlock
import torch
import torch.nn as nn
from models.base import BaseODModel
import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from fastdtw import fastdtw
from data.loader import _read
from data.temporal import steps_per_day, training_end


class STGODE(BaseODModel):
    """
    Reference code: https://github.com/square-coder/STGODE
    """

    def __init__(self, A_sp, A_se, **args):
        super(STGODE, self).__init__(**args)
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    STGCNBlock(
                        in_channels=self.node_num,
                        out_channels=[64, 32, 64],
                        node_num=self.node_num,
                        A_hat=A_sp,
                        seq_len=self.seq_len,
                    ),
                    STGCNBlock(
                        in_channels=64,
                        out_channels=[64, 32, 64],
                        node_num=self.node_num,
                        A_hat=A_sp,
                        seq_len=self.seq_len,
                    ),
                )
                for _ in range(3)
            ]
        )

        # semantic graph
        self.se_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    STGCNBlock(
                        in_channels=self.node_num,
                        out_channels=[64, 32, 64],
                        node_num=self.node_num,
                        A_hat=A_se,
                        seq_len=self.seq_len,
                    ),
                    STGCNBlock(
                        in_channels=64,
                        out_channels=[64, 32, 64],
                        node_num=self.node_num,
                        A_hat=A_se,
                        seq_len=self.seq_len,
                    ),
                )
                for _ in range(3)
            ]
        )

        self.pred = nn.Sequential(
            nn.Linear(self.seq_len * 64, self.horizon * 32),
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon * self.output_dim),
        )

    def forward_single(self, x, label=None):  # (b, t, n, f)
        b = x.shape[0]
        x = x.transpose(1, 2)
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        n = x.shape[1]
        x = x.reshape((b, n, -1))
        x = self.pred(x)  # (b, n, horizon * output_dim)
        x = x.view(b, n, self.horizon, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (b, horizon, n, output_dim)

        return x


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    sp_matrix = adj_mx + np.transpose(adj_mx)
    sp_matrix = _normalize_adj_mx(sp_matrix).to(device)
    se_matrix = _construct_se_matrix(data_path, config)
    se_matrix = _normalize_adj_mx(se_matrix).to(device)
    return dict(A_sp=sp_matrix, A_se=se_matrix)


def build_model(config, node_num, **ctx):
    return STGODE(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        A_sp=ctx["A_sp"],
        A_se=ctx["A_se"],
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
    )


def _construct_se_matrix(data_path, config):
    folder, series, scaler = _read(data_path, config)
    end = training_end(folder, config.data.horizon)
    # Training-only production profiles; inverse before aggregating counts.
    data = scaler.inverse_transform(series[:end], device="cpu").sum(axis=(2, 3))
    sample_num, node_num = data.shape
    period = config.model.params.tpd or steps_per_day(config.data.frequency)
    if sample_num < period:
        raise ValueError("STGODE requires at least one training day to build its semantic graph")
    data_mean = np.mean(
        [
            data[period * i : period * (i + 1)]
            for i in range(sample_num // period)
        ],
        axis=0,
    )
    data_mean = data_mean.T
    dist_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i, node_num):
            dist = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / max(std, 1e-8)
    dist_matrix = np.exp(-(dist_matrix**2) / config.model.params.sigma**2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > config.model.params.thres] = 1
    return dtw_matrix


def _normalize_adj_mx(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 0.0001] = 0.0001
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), adj_mx), diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        od=True,
        od_cqr=True,
        setup=setup,
    )
