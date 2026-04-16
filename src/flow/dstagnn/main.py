import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch
from base.runner import run_experiment
from dstagnn_model import DSTAGNN
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx, calculate_cheb_poly


def add_args(parser):
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--nb_block", type=int, default=2)
    parser.add_argument("--nb_chev_filter", type=int, default=16)
    parser.add_argument("--nb_time_filter", type=int, default=16)
    parser.add_argument("--time_stride", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_k", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--cheb_mask_rank", type=int, default=8)
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


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
    adj_tensor = torch.tensor(adj).to(device)
    return {"cheb_poly": cheb_poly, "adj_pa": adj_tensor}


def build_model(args, node_num, **ctx):
    return DSTAGNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=args.device,
        cheb_poly=ctx["cheb_poly"],
        order=args.order,
        nb_block=args.nb_block,
        nb_chev_filter=args.nb_chev_filter,
        nb_time_filter=args.nb_time_filter,
        time_stride=args.time_stride,
        adj_pa=ctx["adj_pa"],
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_k,
        n_head=args.n_head,
        horizon=args.horizon,
        seq_len=args.seq_len,
        mask_rank=args.cheb_mask_rank,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="DSTAGNN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        init_weights=True,
    )
