import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx

try:
    from .fmgcn3_model import FMGCN3
except ImportError:
    from fmgcn3_model import FMGCN3


def add_args(parser):
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--graph_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--diffusion_order", type=int, default=2)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--highway_window", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    supports = normalize_adj_mx(adj_mx, args.adj_type)
    support_priors = None
    if supports:
        support_priors = [
            torch.tensor(s, dtype=torch.float32, device=device) for s in supports
        ]
        support_priors = [
            s / s.sum(dim=-1, keepdim=True).clamp_min(1e-6) for s in support_priors
        ]
    return {"support_priors": support_priors}


def build_model(args, node_num, **ctx):
    return FMGCN3(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        graph_dim=args.graph_dim,
        num_layers=args.num_layers,
        diffusion_order=args.diffusion_order,
        mlp_hidden=args.mlp_hidden,
        highway_window=args.highway_window,
        dropout=args.dropout,
        support_priors=ctx.get("support_priors"),
        seq_len=args.seq_len,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="FMGCN3",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        init_weights=True,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.MultiStepLR(
            o, milestones=[1, 30, 60, 90], gamma=0.5
        ),
    )
