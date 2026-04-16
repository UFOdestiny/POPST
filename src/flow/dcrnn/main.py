import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from dcrnn_model import DCRNN
from dcrnn_engine import DCRNN_Engine, DCRNN_Engine_Quantile
from utils.dataloader import load_adj_from_numpy


def add_args(parser):
    parser.add_argument("--n_filters", type=int, default=16)
    parser.add_argument("--max_diffusion_step", type=int, default=2)
    parser.add_argument("--filter_type", type=str, default="doubletransition")
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--cl_decay_steps", type=int, default=2000)
    parser.add_argument("--lrate", type=float, default=1e-2)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    return {"adj_mx": adj_mx}


def build_model(args, node_num, **ctx):
    device = torch.device("cuda:0")
    return DCRNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=device,
        adj_mx=ctx["adj_mx"],
        n_filters=args.n_filters,
        max_diffusion_step=args.max_diffusion_step,
        filter_type=args.filter_type,
        num_rnn_layers=args.num_rnn_layers,
        cl_decay_steps=args.cl_decay_steps,
        horizon=args.horizon,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="DCRNN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        engine_cls=DCRNN_Engine,
        engine_quantile_cls=DCRNN_Engine_Quantile,
        device_override="cuda:0",
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[10, 50, 90], gamma=0.1),
    )
