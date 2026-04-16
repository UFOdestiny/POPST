import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../../../'))

import torch
from base.runner import run_experiment
from gmel_model import GMEL
from utils.dataloader import load_adj_from_numpy


def add_args(parser):
    parser.add_argument("--in_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--num_hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=6)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    return dict(adj_mx=adj_mx)


def build_model(args, node_num, **ctx):
    return GMEL(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        g=ctx["adj_mx"],
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        num_hidden=args.num_hidden,
        num_layers=args.layers,
        num_heads=args.heads,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="GMEL",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MSE",
        setup=setup,
        make_optimizer=lambda m, a: torch.optim.Adam(m.parameters()),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
