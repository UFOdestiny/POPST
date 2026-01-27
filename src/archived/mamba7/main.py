import os
import numpy as np
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine

import torch

torch.set_num_threads(8)

from src.flow.mamba7.mamba_model import UNetMamba
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.log import get_logger
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.graph_algo import normalize_adj_mx


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--graph_embed_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_graph_layers", type=int, default=1)
    parser.add_argument("--gate_init", type=float, default=-2.0)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    args = parser.parse_args()

    args.model_name = "Mamba7"
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
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, "scalap")[0]
    gso = torch.tensor(gso).to(device)

    dataloader, scaler = load_dataset(data_path, args, logger)
    args, engine_template = check_quantile(args, BaseEngine, CQR_Engine)
    model = UNetMamba(
        node_num=node_num,
        input_dim=args.seq_len,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        num_layers=args.num_layers,
        d_model=args.d_model,
        feature=args.feature,
        adj=gso,
        graph_embed_dim=args.graph_embed_dim,
        dropout=args.dropout,
        num_graph_layers=args.num_graph_layers,
        gate_init=args.gate_init,
    )

    loss_fn = "MAE"
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
