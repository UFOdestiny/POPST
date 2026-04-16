import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from gwnet_model import GWNET
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--adp_adj", type=int, default=1)
    parser.add_argument("--init_dim", type=int, default=32)
    parser.add_argument("--skip_dim", type=int, default=256)
    parser.add_argument("--end_dim", type=int, default=512)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    return {"supports": supports}


def build_model(args, node_num, **ctx):
    return GWNET(
        node_num=node_num,
        supports=ctx["supports"],
        adp_adj=args.adp_adj,
        dropout=args.dropout,
        residual_channels=args.init_dim,
        dilation_channels=args.init_dim,
        skip_channels=args.skip_dim,
        end_channels=args.end_dim,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        horizon=args.horizon,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="GWNET",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
