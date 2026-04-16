import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch
from base.runner import run_experiment
from stgode_model import STGODE
from utils.dataloader import load_adj_from_numpy
from fastdtw import fastdtw


def add_args(parser):
    parser.add_argument("--tpd", type=int, default=96, help="time per day")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--thres", type=float, default=0.6)
    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def _normalize_adj_mx(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(
        np.multiply(diag.reshape((-1, 1)), adj_mx), diag.reshape((1, -1))
    )
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def _construct_se_matrix(data_path, args):
    ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
    data = ptr["data"][..., 0]
    sample_num, node_num = data.shape

    data_mean = np.mean(
        [
            data[args.tpd * i : args.tpd * (i + 1)]
            for i in range(sample_num // args.tpd)
        ],
        axis=0,
    )
    data_mean = data_mean.T

    dist_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i, node_num):
            dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    for i in range(node_num):
        for j in range(i):
            dist_matrix[i][j] = dist_matrix[j][i]

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    dist_matrix = np.exp(-(dist_matrix**2) / args.sigma**2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres] = 1
    return dtw_matrix


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    sp_matrix = adj_mx + np.transpose(adj_mx)
    sp_matrix = _normalize_adj_mx(sp_matrix).to(device)

    se_matrix = _construct_se_matrix(data_path, args)
    se_matrix = _normalize_adj_mx(se_matrix).to(device)
    return {"A_sp": sp_matrix, "A_se": se_matrix}


def build_model(args, node_num, **ctx):
    return STGODE(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        A_sp=ctx["A_sp"],
        A_se=ctx["A_se"],
        horizon=args.horizon,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="STGODE",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        make_optimizer=lambda m, a: torch.optim.AdamW(m.parameters(), lr=a.lrate, weight_decay=a.wdecay),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=20, gamma=0.5),
    )
