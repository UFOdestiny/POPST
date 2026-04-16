import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch
from base.runner import run_experiment
from uqgnn_model import UQGNN
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--rank_s", type=int, default=512)
    parser.add_argument("--rank_t", type=int, default=512)
    parser.add_argument("--hidden_dim_s", type=int, default=64)
    parser.add_argument("--hidden_dim_t", type=int, default=64)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip_grad_norm", type=float, default=0)
    parser.add_argument("--min_vec", type=float, default=1e-6)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    return {"gso": gso}


def build_model(args, node_num, **ctx):
    device = torch.device("cuda:0")
    return UQGNN(
        A=ctx["gso"],
        seq_len=args.seq_len,
        node_num=node_num,
        hidden_dim_t=args.hidden_dim_t,
        hidden_dim_s=args.hidden_dim_s,
        rank_t=args.rank_t,
        rank_s=args.rank_s,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.horizon,
        device=device,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        min_vec=args.min_vec,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="UQGNN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        device_override="cuda:0",
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
