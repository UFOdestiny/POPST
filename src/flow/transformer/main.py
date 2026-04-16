import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from src.flow.transformer.transformer_model import myTransformer


def add_args(parser):
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)
    parser.set_defaults(bs=16)


def build_model(args, node_num, **ctx):
    return myTransformer(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        num_layers=args.num_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        feature=args.input_dim,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="Transformer",
        add_args=add_args,
        build_model=build_model,
        make_optimizer=lambda m, a: torch.optim.Adam(m.parameters()),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
