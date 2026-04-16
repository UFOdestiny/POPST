import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from dgcrn_model import DGCRN
from dgcrn_engine import DGCRN_Engine, DGCRN_Engine_Quantile
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--gcn_depth", type=int, default=3)
    parser.add_argument("--rnn_size", type=int, default=64)
    parser.add_argument("--hyperGNN_dim", type=int, default=16)
    parser.add_argument("--node_dim", type=int, default=32)
    parser.add_argument("--middle_dim", type=int, default=2)
    parser.add_argument("--tanhalpha", type=int, default=3)
    parser.add_argument("--cl_decay_step", type=int, default=2000)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--tpd", type=int, default=24)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, "doubletransition")
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    return {"supports": supports, "device": device}


def build_model(args, node_num, **ctx):
    return DGCRN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=ctx["device"],
        predefined_adj=ctx["supports"],
        gcn_depth=args.gcn_depth,
        rnn_size=args.rnn_size,
        hyperGNN_dim=args.hyperGNN_dim,
        node_dim=args.node_dim,
        middle_dim=args.middle_dim,
        list_weight=[0.05, 0.95, 0.95],
        tpd=args.tpd,
        tanhalpha=args.tanhalpha,
        cl_decay_step=args.cl_decay_step,
        dropout=args.dropout,
        horizon=args.horizon,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="DGCRN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        engine_cls=DGCRN_Engine,
        engine_quantile_cls=DGCRN_Engine_Quantile,
    )
