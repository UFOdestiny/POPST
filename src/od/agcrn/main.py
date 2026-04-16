import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from base.runner import run_experiment
from agcrn_model import AGCRN


def add_args(parser):
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--rnn_unit", type=int, default=24)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def build_model(args, node_num, **ctx):
    return AGCRN(
        node_num=node_num,
        input_dim=node_num,
        output_dim=node_num,
        embed_dim=args.embed_dim,
        rnn_unit=args.rnn_unit,
        num_layer=args.num_layer,
        cheb_k=args.cheb_k,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="AGCRN_OD",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MSE",
        init_weights=True,
    )
