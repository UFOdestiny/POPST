import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from src.flow.mamba4.mamba_model import UNetMamba


def add_args(parser):
    parser.add_argument("--num_layers", type=int, default=3, help="ST-Mamba块数量")
    parser.add_argument("--d_model", type=int, default=96, help="模型维度")
    parser.add_argument("--num_heads", type=int, default=8, help="空间注意力头数")
    parser.add_argument("--d_ff", type=int, default=256, help="FFN/MoE中间维度")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout率")
    parser.add_argument("--num_experts", type=int, default=4, help="MoE专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="MoE激活的专家数")
    parser.add_argument("--window_size", type=int, default=4, help="滑动窗口大小")
    parser.add_argument("--d_state", type=int, default=16, help="Mamba SSM状态维度")
    parser.add_argument("--d_conv", type=int, default=4, help="Mamba局部卷积宽度")
    parser.add_argument("--expand", type=int, default=2, help="Mamba内部扩展因子")
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def build_model(args, node_num, **ctx):
    return UNetMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        top_k=args.top_k,
        window_size=args.window_size,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="Mamba4",
        add_args=add_args,
        build_model=build_model,
        make_optimizer=lambda m, a: torch.optim.AdamW(m.parameters(), lr=a.lrate, weight_decay=a.wdecay),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.max_epochs, eta_min=1e-6),
    )
