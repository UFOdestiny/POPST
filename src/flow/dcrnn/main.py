import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

torch.set_num_threads(3)

from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine
from dcrnn_model import DCRNN
from dcrnn_engine import DCRNN_Engine,DCRNN_Engine_Quantile
from utils.args import check_quantile, get_public_config, get_log_path, print_args
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from base.metrics import masked_mae
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--n_filters", type=int, default=16)
    parser.add_argument("--max_diffusion_step", type=int, default=2)
    parser.add_argument("--filter_type", type=str, default="doubletransition")
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--cl_decay_steps", type=int, default=2000)

    parser.add_argument("--lrate", type=float, default=1e-2)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()
    args.model_name = "DCRNN"
    if args.quantile:
        args.model_name += "_CQR"
    log_dir = get_log_path(args)
    logger = get_logger(
        log_dir,
        __name__,
    )
    print_args(logger, args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(0)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    # logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)

    dataloader, scaler = load_dataset(data_path, args, logger)
    args, engine_template = check_quantile(args, DCRNN_Engine, DCRNN_Engine_Quantile)
    if args.quantile:
        args.output_dim=3
        args.horizon=1
        # args.input_dim=1

    model = DCRNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=device,
        adj_mx=adj_mx,
        n_filters=args.n_filters,
        max_diffusion_step=args.max_diffusion_step,
        filter_type=args.filter_type,
        num_rnn_layers=args.num_rnn_layers,
        cl_decay_steps=args.cl_decay_steps,
        horizon=args.horizon,
        seq_len=args.seq_len
    )

    loss_fn = "MAE"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    steps = [10, 50, 90]  # CA: [5, 50, 90], others: [10, 50, 90]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=steps, gamma=0.1
    )
    
    engine = engine_template(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        normalize=args.normalize,
        alpha=args.quantile_alpha,
        metric_list=["MAE", "MAPE", "RMSE"],

        args=args,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path, args.export)


if __name__ == "__main__":
    main()
