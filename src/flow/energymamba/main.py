import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch

from base.runner import run_experiment
from src.flow.energymamba.ACQR_engine import ACQR_Engine
from src.flow.energymamba.em_model import EnergyMamba
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    # GE-Mamba architecture
    parser.add_argument("--num_layers", type=int, default=2)   # blocks per stage (K)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--unet_depth", type=int, default=2)   # encoder stages (S)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # AS-CQR calibration (used only under --cqr)
    parser.add_argument("--acqr_window", type=int, default=100)
    parser.add_argument("--acqr_gamma", type=float, default=0.005)
    parser.add_argument("--acqr_delta", type=float, default=1e-6)

    # optimisation
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)


def setup(args, data_path, adj_path, node_num, device, logger):
    # Symmetric normalised adjacency with self-loops: D̃^{-1/2}(A+I)D̃^{-1/2}.
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)              # drop any pre-existing self-loops
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]      # adds self-loops + sym-normalises
    return {"adj": np.asarray(gso, dtype=np.float32)}


def build_model(args, node_num, **ctx):
    return EnergyMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        adj=ctx["adj"],
        num_layers=args.num_layers,
        d_model=args.d_model,
        depth=args.unet_depth,
        d_state=args.d_state,
        expand=args.expand,
        d_conv=args.d_conv,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="EnergyMamba",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        engine_quantile_cls=ACQR_Engine,
        device_override="cuda:0",
        make_optimizer=lambda m, a: torch.optim.AdamW(
            m.parameters(), lr=a.lrate, weight_decay=a.wdecay
        ),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(
            o, step_size=a.step_size, gamma=a.gamma
        ),
    )
