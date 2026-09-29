import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch

from base.runner import run_experiment
from pdr_reg_model import PDRReg
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--zone_embed_dim", type=int, default=16)
    parser.add_argument("--pdr_num_spatial_layers", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--head_hidden_dim", type=int, default=128)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=2e-4)
    parser.add_argument("--wdecay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    return {"gso": gso}


def build_model(args, node_num, **ctx):
    return PDRReg(
        A=ctx["gso"],
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        context_dim=args.context_dim,
        zone_embed_dim=args.zone_embed_dim,
        num_spatial_layers=args.pdr_num_spatial_layers,
        num_experts=args.num_experts,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="PDR_REG",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MAE",
        metric_list=["MAE", "MAPE", "MSE", "RMSE"],
        od=True,
        od_cqr=True,
        setup=setup,
        make_scheduler=lambda optimizer, args: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        ),
    )
