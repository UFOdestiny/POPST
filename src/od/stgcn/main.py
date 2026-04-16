import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch
from base.runner import run_experiment
from stgcn_model import STGCN_OD
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--Kt", type=int, default=2)
    parser.add_argument("--Ks", type=int, default=3)
    parser.add_argument("--block_num", type=int, default=2)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def setup(args, data_path, adj_path, node_num, device, logger):
    args.input_dim = node_num
    args.output_dim = node_num

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, "scalap")[0]
    gso = torch.tensor(gso).to(device)

    Ko = args.seq_len - (args.Kt - 1) * 2 * args.block_num
    blocks = []
    blocks.append([args.input_dim])
    for l in range(args.block_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args.input_dim])

    return dict(gso=gso, blocks=blocks)


def build_model(args, node_num, **ctx):
    return STGCN_OD(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        gso=ctx["gso"],
        blocks=ctx["blocks"],
        Kt=args.Kt,
        Ks=args.Ks,
        dropout=args.dropout,
        feature=args.input_dim,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="STGCN_OD",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MSE",
        setup=setup,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
        device_override="cuda:0",
    )
