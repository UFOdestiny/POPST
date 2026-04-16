import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from gluonts_model import GluonTSModel


def add_args(parser):
    parser.add_argument("--init_dim", type=int, default=32, help="初始投影维度")
    parser.add_argument("--hid_dim", type=int, default=32, help="隐藏层维度")
    parser.add_argument("--end_dim", type=int, default=64, help="输出层维度")
    parser.add_argument("--num_layers", type=int, default=3, help="TCN层数")
    parser.add_argument("--kernel_size", type=int, default=3, help="卷积核大小")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--use_attention", type=bool, default=True, help="是否使用注意力")
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def build_model(args, node_num, **ctx):
    return GluonTSModel(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        init_dim=args.init_dim,
        hid_dim=args.hid_dim,
        end_dim=args.end_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        num_heads=args.num_heads,
        use_attention=args.use_attention,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="GluonTS",
        add_args=add_args,
        build_model=build_model,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
