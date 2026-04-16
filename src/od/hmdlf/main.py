import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../../../'))

import torch
from base.runner import run_experiment
from hmdlf_model import HMDLF


def add_args(parser):
    parser.add_argument("--cnn_out", type=int, default=8)
    parser.add_argument("--gru_hidden", type=int, default=16)
    parser.add_argument("--predictor_hidden", type=int, default=32)
    parser.add_argument("--share_encoder", type=bool, default=False)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def build_model(args, node_num, **ctx):
    return HMDLF(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        cnn_out=args.cnn_out,
        gru_hidden=args.gru_hidden,
        predictor_hidden=args.predictor_hidden,
        num_mobility=node_num,
        share_encoder=args.share_encoder,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="HMDLF",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MSE",
        make_optimizer=lambda m, a: torch.optim.Adam(m.parameters()),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(o, step_size=a.step_size, gamma=a.gamma),
    )
