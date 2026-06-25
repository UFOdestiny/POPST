import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from base.runner import run_experiment, NO_OPTIMIZER
from base.engine import BaseEngine_OD_Stat
from var_model import VAR
from utils.dataloader import load_dataset_plain


def add_args(parser):
    parser.add_argument("--k", type=int, default=6, help="number of low-rank temporal factors / VAR lag order")
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def build_model(args, node_num, **ctx):
    return VAR(
        k=args.k,
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="VAR",
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
