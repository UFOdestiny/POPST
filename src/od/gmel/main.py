import os
import numpy as np
import sys

sys.path.append(os.path.abspath(__file__ + '/../../../../'))
sys.path.append("/home/dy23a.fsu/st/")

from base.engine import BaseEngine
from base.quantile_engine import Quantile_Engine

import torch

torch.set_num_threads(8)

from gmel_model import GMEL
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_adj_from_numpy, load_dataset, get_dataset_info
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    parser.add_argument("--in_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--num_hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=6)

    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    args = parser.parse_args()

    args.model_name = "GMEL"
    # args.bs = 64
    log_dir = get_log_path(args)
    logger = get_logger(
        log_dir,
        __name__,
    )
    print_args(logger, args)  # logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    adj_mx = load_adj_from_numpy(adj_path)

    dataloader, scaler = load_dataset(data_path, args, logger)
    args, engine_template = check_quantile(args, BaseEngine, Quantile_Engine)
    model = GMEL(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        g=adj_mx,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        num_hidden=args.num_hidden,
        num_layers=args.layers,
        num_heads=args.heads,
    )

    loss_fn = "MSE"  # masked_mae
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
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
        clip_grad_value=0,
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
