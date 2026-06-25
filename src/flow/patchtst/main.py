import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from patchtst_model import PatchTST


def add_args(parser):
    parser.add_argument("--patch_len", type=int, default=2, help="patch length")
    parser.add_argument("--stride", type=int, default=1, help="patch sliding stride")
    parser.add_argument("--d_model", type=int, default=64, help="model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=64, help="feed-forward network dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="number of Transformer layers")
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def build_model(args, node_num, **ctx):
    return PatchTST(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="PatchTST",
        add_args=add_args,
        build_model=build_model,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
