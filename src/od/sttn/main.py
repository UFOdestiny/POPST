import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../../../'))

import numpy as np
import torch
from base.runner import run_experiment
from sttn_model import STTN
from sttn_engine import STTN_Engine
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--rank_s", type=int, default=2)
    parser.add_argument("--rank_t", type=int, default=2)
    parser.add_argument("--hidden_dim_s", type=int, default=2)
    parser.add_argument("--hidden_dim_t", type=int, default=2)
    parser.add_argument('--min_vec', type=float, default=1e-3)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5)
    parser.set_defaults(bs=8)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    return dict(gso=gso, device=device)


def build_model(args, node_num, **ctx):
    return STTN(
        A=ctx["gso"],
        seq_len=args.seq_len,
        node_num=node_num,
        hidden_dim_t=args.hidden_dim_t,
        hidden_dim_s=args.hidden_dim_s,
        rank_t=args.rank_t,
        rank_s=args.rank_s,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.horizon,
        device=ctx["device"],
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        min_vec=args.min_vec,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="STTN",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MGAU",
        engine_cls=STTN_Engine,
        setup=setup,
        metric_list=["MSE", "MAE", "MAPE", "RMSE"],
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
