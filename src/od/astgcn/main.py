import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch
from base.runner import run_experiment
from astgcn_model import ASTGCN
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx, calculate_cheb_poly


def add_args(parser):
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--nb_block", type=int, default=2)
    parser.add_argument("--nb_chev_filter", type=int, default=64)
    parser.add_argument("--nb_time_filter", type=int, default=64)
    parser.add_argument("--time_stride", type=int, default=1)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    adj = np.zeros((node_num, node_num), dtype=np.float32)
    for n in range(node_num):
        idx = np.nonzero(adj_mx[n])[0]
        adj[n, idx] = 1

    L_tilde = normalize_adj_mx(adj, "scalap")[0]
    cheb_poly = [
        torch.from_numpy(i).type(torch.FloatTensor).to(device)
        for i in calculate_cheb_poly(L_tilde, args.order)
    ]
    return dict(cheb_poly=cheb_poly)


def build_model(args, node_num, **ctx):
    return ASTGCN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        horizon=node_num,
        device=args.device,
        cheb_poly=ctx["cheb_poly"],
        order=args.order,
        nb_block=args.nb_block,
        nb_chev_filter=args.nb_chev_filter,
        nb_time_filter=args.nb_time_filter,
        time_stride=args.time_stride,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="ASTGCN_OD",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MSE",
        init_weights=True,
        setup=setup,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.max_epochs, eta_min=1e-6),
    )
