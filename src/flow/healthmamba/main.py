import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch

from base.runner import run_experiment
from src.flow.healthmamba.PHQC_engine import PHQC_Engine
from src.flow.healthmamba.hm_model import HealthMamba
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    # GraphMamba architecture
    parser.add_argument("--num_layers", type=int, default=2)   # G-Mamba blocks per stage
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_hid", type=int, default=128)
    parser.add_argument("--unet_depth", type=int, default=2)   # encoder stages S
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=32)     # adaptive-graph emb
    parser.add_argument("--dropout", type=float, default=0.3)

    # Ablation toggles (RQ3). 1 = on, 0 = off.
    parser.add_argument("--use_stce", type=int, default=1)
    parser.add_argument("--use_gmamba", type=int, default=1)
    parser.add_argument("--use_node", type=int, default=1)
    parser.add_argument("--use_dist", type=int, default=1)
    parser.add_argument("--use_param", type=int, default=1)

    # Post-hoc quantile calibration (PHQC_Engine).
    parser.add_argument(
        "--phqc", type=str, default="horizon", choices=["horizon", "global"],
        help="post-hoc calibration margin granularity",
    )
    parser.add_argument("--phqc_mc", type=int, default=20)     # MC-dropout passes M
    parser.add_argument("--phqc_w_nll", type=float, default=1.0)
    parser.add_argument("--phqc_w_param", type=float, default=1.0)
    parser.add_argument("--phqc_w_calib", type=float, default=1.0)

    # optimisation
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)


def setup(args, data_path, adj_path, node_num, device, logger):
    # Symmetric normalised prior adjacency with self-loops: D̃^{-1/2}(A+I)D̃^{-1/2}.
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)              # drop any pre-existing self-loops
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]      # adds self-loops + sym-normalises
    return {"adj": np.asarray(gso, dtype=np.float32)}


def build_model(args, node_num, **ctx):
    return HealthMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        adj=ctx["adj"],
        # Under --cqr the runner widens output_dim to 3*F; cqr_channels carries
        # the true feature count F and also switches the model into UQ mode.
        cqr_channels=getattr(args, "cqr_channels", None),
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_hid=args.d_hid,
        depth=args.unet_depth,
        d_state=args.d_state,
        expand=args.expand,
        d_conv=args.d_conv,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        use_stce=bool(args.use_stce),
        use_gmamba=bool(args.use_gmamba),
        use_node=bool(args.use_node),
        use_dist=bool(args.use_dist),
        use_param=bool(args.use_param),
    )


if __name__ == "__main__":
    run_experiment(
        model_name="HealthMamba",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        # PHQC (uncertainty-aware) engine is selected only under --cqr, exactly
        # like ACQR/SCQR for energymamba/trustenergy.  Default (--cqr no) runs
        # the point regressor on BaseEngine.
        engine_quantile_cls=PHQC_Engine,
        device_override="cuda:0",
        make_optimizer=lambda m, a: torch.optim.AdamW(
            m.parameters(), lr=a.lrate, weight_decay=a.wdecay
        ),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(
            o, step_size=a.step_size, gamma=a.gamma
        ),
    )
