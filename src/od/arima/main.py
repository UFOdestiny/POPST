import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from base.runner import run_experiment, NO_OPTIMIZER
from base.engine import BaseEngine_OD_Stat
from arima_model import ARIMA_
from utils.args import tuple_type
from utils.dataloader import load_dataset_plain


def add_args(parser):
    parser.add_argument("--order", type=tuple_type, default=(6, 0, 0))
    parser.add_argument("--n_threads", type=int, default=16)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def build_model(args, node_num, **ctx):
    return ARIMA_(
        order=args.order,
        n_threads=args.n_threads,
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="ARIMA",
        add_args=add_args,
        build_model=build_model,
        loss_fn="MAE",
        od=True,
        engine_cls=BaseEngine_OD_Stat,
        make_optimizer=NO_OPTIMIZER,
        load_data=load_dataset_plain,
        device_override="cpu",
        train_with_export=True,
    )
