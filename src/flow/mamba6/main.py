import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from src.flow.mamba6.mamba_model import UNetMamba
from utils.dataloader import load_adj_from_numpy


def add_args(parser):
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--sample_factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ffn_expand", type=int, default=2)
    parser.add_argument("--use_multiscale", type=bool, default=True)
    parser.add_argument("--use_temporal_conv", type=bool, default=True)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    return {"adj_mx": adj_mx}


def build_model(args, node_num, **ctx):
    return UNetMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        num_layers=args.num_layers,
        d_model=args.d_model,
        feature=args.input_dim,
        sample_factor=args.sample_factor,
        dropout=args.dropout,
        ffn_expand=args.ffn_expand,
        use_multiscale=args.use_multiscale,
        use_temporal_conv=args.use_temporal_conv,
        adj_mx=ctx["adj_mx"],
    )


if __name__ == "__main__":
    run_experiment(
        model_name="Mamba6",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        make_optimizer=lambda m, a: torch.optim.Adam(m.parameters()),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
