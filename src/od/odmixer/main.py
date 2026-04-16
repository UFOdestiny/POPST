import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from odmixer_model import ODMixer


def add_args(parser):
    parser.add_argument("--hid_dim", type=int, default=8)
    parser.add_argument("--layer_nums", type=int, default=3)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def setup(args, data_path, adj_path, node_num, device, logger):
    args.input_dim = node_num
    args.output_dim = node_num


def build_model(args, node_num, **ctx):
    return ODMixer(
        node_num=node_num,
        input_dim=node_num,
        output_dim=args.output_dim,
        dropout=args.dropout,
        feature=args.input_dim,
        horizon=args.horizon,
        seq_len=args.seq_len,
        hid_dim=args.hid_dim,
        layer_nums=args.layer_nums,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="ODMixer",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MSE",
        setup=setup,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
        device_override="cuda:0",
    )
