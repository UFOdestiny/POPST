import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from sttn_model import STTN
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--mlp_expand", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--end_dim", type=int, default=512)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    return {"supports": supports, "device": device}


def build_model(args, node_num, **ctx):
    return STTN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=ctx["device"],
        supports=ctx["supports"],
        blocks=args.blocks,
        mlp_expand=args.mlp_expand,
        hidden_channels=args.hid_dim,
        end_channels=args.end_dim,
        dropout=args.dropout,
        horizon=args.horizon,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="STTN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
    )
