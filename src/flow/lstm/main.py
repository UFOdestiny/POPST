import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from lstm_model import LSTM


def add_args(parser):
    parser.add_argument("--init_dim", type=int, default=32)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--end_dim", type=int, default=512)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def build_model(args, node_num, **ctx):
    return LSTM(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        init_dim=args.init_dim,
        hid_dim=args.hid_dim,
        end_dim=args.end_dim,
        layer=args.layer,
        dropout=args.dropout,
        horizon=args.horizon,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="LSTM",
        add_args=add_args,
        build_model=build_model,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
