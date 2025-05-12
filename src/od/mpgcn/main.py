import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))
sys.path.append("/home/dy23a.fsu/st/")
sys.path.append("/home/ec2-user/POPST")
from od.mpgcn.mpgcn_engine import MPGCN_Engine
from base.quantile_engine import Quantile_Engine
import torch
import numpy as np

torch.set_num_threads(8)

from mpgcn_model import GCN, MPGCN, Adj_Processor
from base.engine import BaseEngine
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import get_dataset_info, load_dataset, load_dataset_MPGCN
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    parser.add_argument("-hidden", "--hidden_dim", type=int, default=8)
    parser.add_argument(
        "-kernel",
        "--kernel_type",
        type=str,
        choices=[
            "chebyshev",
            "localpool",
            "random_walk_diffusion",
            "dual_random_walk_diffusion",
        ],
        default="localpool",
    )  # GCN kernel type
    parser.add_argument("--cheby_order", type=int, default=1)  # GCN chebyshev order
    parser.add_argument("--K", type=int, default=1)

    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()

    # args.bs = 32  # test

    args.model_name = "MPGCN_OD"
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

    dataloader, scaler = load_dataset_MPGCN(data_path, args, logger)

    args, engine_template = check_quantile(args, MPGCN_Engine, Quantile_Engine)

    model = MPGCN(
        M=2,  # 2 branches: one for adj; the other for dynamic O/G cosine correlation graph
        K=args.K,
        G=np.load(adj_path),
        input_dim=1,
        output_dim=1,
        lstm_hidden_dim=args.hidden_dim,
        lstm_num_layers=1,
        gcn_hidden_dim=args.hidden_dim,
        gcn_num_layers=3,
        num_nodes=node_num,
        user_bias=True,
        kernel_type=args.kernel_type,
        cheby_order=args.cheby_order,
    )

    loss_fn = "MSE"
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
        metric_list=["MAE", "MAPE", "RMSE"],
        args=args,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path, args.export)


if __name__ == "__main__":
    main()
