import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch

from base.runner import run_experiment
from pdr_reg_model import PDRReg
from zero_cqr_engine import ZeroCQREngine
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
    parser.add_argument("--zero_cqr_alpha", type=float, default=0.1)
    parser.add_argument("--zero_cqr_gate_quantile", type=float, default=0.95)
    parser.add_argument("--zero_cqr_grid_size", type=int, default=33)
    parser.add_argument("--zero_cqr_min_group", type=int, default=64)
    parser.add_argument(
        "--zero_cqr_mse_weight", type=float, default=1.0,
        help="Relative validation-MSE weight when selecting the sparse zero gate.",
    )
    parser.add_argument("--zero_cqr_aux_epochs", type=int, default=4)
    parser.add_argument("--zero_cqr_aux_samples", type=int, default=400000)
    parser.add_argument("--zero_cqr_period", type=int, default=96)
    parser.add_argument(
        "--zero_cqr_disable_online", action="store_true",
        help="Disable ZeroCQR's causal online residual correction at test time.",
    )
    parser.add_argument(
        "--zero_cqr_zero_floor", type=float, default=1e-3,
        help="Set non-negative post-hoc point forecasts at or below this value to zero.",
    )
    parser.add_argument("--zero_cqr_active_bins", type=int, default=8)


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
        model_name="PDR_REG_POST",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MAE",
        metric_list=ZeroCQREngine.DEFAULT_METRICS,
        od=True,
        od_cqr=True,
        engine_cls=ZeroCQREngine,
        engine_extras=lambda args: {
            "zero_cqr_alpha": args.zero_cqr_alpha,
            "zero_cqr_gate_quantile": args.zero_cqr_gate_quantile,
            "zero_cqr_grid_size": args.zero_cqr_grid_size,
            "zero_cqr_min_group": args.zero_cqr_min_group,
            "zero_cqr_mse_weight": args.zero_cqr_mse_weight,
            "zero_cqr_aux_epochs": args.zero_cqr_aux_epochs,
            "zero_cqr_aux_samples": args.zero_cqr_aux_samples,
            "zero_cqr_period": args.zero_cqr_period,
            "zero_cqr_enable_online": not args.zero_cqr_disable_online,
            "zero_cqr_zero_floor": args.zero_cqr_zero_floor,
            "zero_cqr_active_bins": args.zero_cqr_active_bins,
        },
        setup=setup,
        make_scheduler=lambda optimizer, args: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        ),
    )
