"""Isolated post-hoc calibration runner for the Chicago PDR no-context ablation.

It deliberately uses a distinct model name/project at invocation time, so the
source checkpoint directory is read-only from the experiment's perspective.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from base.runner import run_experiment
from src.od.pdr.ablation_runner import add_args
from src.od.pdr_no_context.pdr_no_context_model import PDR_no_context
from src.od.pdr_reg_post.zero_cqr_engine import ZeroCQREngine
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_compare_args(parser):
    add_args(parser)
    parser.add_argument("--calibration_engine", choices=["od_cqr", "zero_cqr"], default="od_cqr")
    parser.add_argument("--zero_cqr_alpha", type=float, default=0.05)
    parser.add_argument("--zero_cqr_active_bins", type=int, default=8)
    parser.add_argument("--zero_cqr_zero_floor", type=float, default=1e-3)
    parser.add_argument("--zero_cqr_aux_epochs", type=int, default=0)
    parser.add_argument("--zero_cqr_period", type=int, default=0)
    parser.add_argument("--zero_cqr_enable_online", action="store_true")


def setup(args, data_path, adj_path, node_num, device, logger):
    adj = load_adj_from_numpy(adj_path) - np.eye(node_num)
    return {"gso": normalize_adj_mx(adj, "uqgnn")[0]}


def build_model(args, node_num, **ctx):
    return PDR_no_context(
        A=ctx["gso"], node_num=node_num, input_dim=args.input_dim,
        output_dim=args.output_dim, seq_len=args.seq_len, horizon=args.horizon,
        context_dim=args.context_dim, zone_embed_dim=args.zone_embed_dim,
        num_spatial_layers=args.pdr_num_spatial_layers, num_experts=args.num_experts,
        head_hidden_dim=args.head_hidden_dim, dropout=args.dropout,
    )


if __name__ == "__main__":
    # Inspect the command line before runner parsing to choose the engine.
    use_zero = "--calibration_engine" in sys.argv and sys.argv[sys.argv.index("--calibration_engine") + 1] == "zero_cqr"
    run_experiment(
        model_name="PDR_no_context_COMPARE",
        add_args=add_compare_args,
        build_model=build_model,
        loss_fn="NLL",
        metric_list=ZeroCQREngine.DEFAULT_METRICS if use_zero else ["MAE", "MAPE", "MSE", "RMSE"],
        od=True,
        od_cqr=True,
        engine_cls=ZeroCQREngine if use_zero else None,
        engine_extras=(lambda args: {
            "zero_cqr_alpha": args.zero_cqr_alpha,
            "zero_cqr_active_bins": args.zero_cqr_active_bins,
            "zero_cqr_zero_floor": args.zero_cqr_zero_floor,
            "zero_cqr_aux_epochs": args.zero_cqr_aux_epochs,
            "zero_cqr_period": args.zero_cqr_period,
            "zero_cqr_enable_online": args.zero_cqr_enable_online,
        }) if use_zero else None,
        setup=setup,
        make_scheduler=lambda optimizer, args: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        ),
    )
