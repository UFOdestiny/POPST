import os
import sys

from src.flow.pgnn.pgnn_engine import PGNN_Engine_Quantile

sys.path.append(os.path.abspath(__file__ + '/../../../../'))

import numpy as np
import torch
from pgnn_engine import PGNN_Engine
from pgnn_model import PGNN
from base.metrics import masked_mae

torch.set_num_threads(3)
from utils.graph_algo import normalize_adj_mx
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    # parser.add_argument('--rank_s', type=int, default=256)
    # parser.add_argument('--rank_t', type=int, default=256)
    # parser.add_argument('--hidden_dim_s', type=int, default=32)
    # parser.add_argument('--hidden_dim_t', type=int, default=16)

    parser.add_argument("--residual_channels", type=int, default=128)
    parser.add_argument("--dilation_channels", type=int, default=128)
    parser.add_argument("--skip_channels", type=int, default=256)
    parser.add_argument("--end_channels", type=int, default=512)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip_grad_value", type=float, default=0)

    # parser.add_argument('--min_vec', type=float, default=1e-5)

    args = parser.parse_args()
    args.model_name = "PGNN"

    log_dir = get_log_path(args)
    logger = get_logger(
        log_dir,
        __name__,
    )
    # logger.info(f"Comment: {args.comment}")

    print_args(logger, args)
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    # device = torch.device(args.device)
    device = torch.device(0)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)

    # logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, "doubletransition")
    supports = [torch.tensor(i).to(device) for i in gso]

    # gso = torch.tensor(gso).to(device)

    dataloader, scaler = load_dataset(data_path, args, logger)

    args, engine_template = check_quantile(args, PGNN_Engine, PGNN_Engine_Quantile)

    model = PGNN(
        num_nodes=node_num,
        cluster_nodes=node_num,
        residual_channels=args.residual_channels,
        dilation_channels=args.dilation_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        supports=supports,
        supports_cluster=supports,
        length=args.seq_len,
        out_dim=args.horizon,
        device=device,
        dropout=args.dropout,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
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
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        normalize=args.normalize,
        alpha=args.quantile_alpha,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path)


if __name__ == "__main__":
    main()
