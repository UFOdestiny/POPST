import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from stllm_model import STLLM


def add_args(parser):
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=384, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of STLLM8 layers")
    parser.add_argument("--reduction_ratio", type=int, default=4, help="Channel recalibration reduction ratio")
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def build_model(args, node_num, **ctx):
    return STLLM(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        reduction_ratio=args.reduction_ratio,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="STLLM8",
        add_args=add_args,
        build_model=build_model,
        make_optimizer=lambda m, a: torch.optim.AdamW(m.parameters(), lr=a.lrate, weight_decay=a.wdecay),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.max_epochs, eta_min=1e-6),
    )
