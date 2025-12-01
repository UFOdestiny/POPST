import os
import numpy as np
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
import torch

torch.set_num_threads(8)

from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine
from d2stgnn_model import D2STGNN
from d2stgnn_engine import D2STGNN_Engine
from utils.args import check_quantile, get_public_config, get_log_path, print_args
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.graph_algo import normalize_adj_mx
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    # parser.add_argument("--num_feat", type=int, default=5)
    parser.add_argument("--num_hidden", type=int, default=32)
    parser.add_argument("--node_hidden", type=int, default=12)
    parser.add_argument("--time_emb_dim", type=int, default=6)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--k_t", type=int, default=3)
    parser.add_argument("--k_s", type=int, default=2)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--cl_epoch", type=int, default=3)
    parser.add_argument("--warm_epoch", type=int, default=30)
    parser.add_argument("--tpd", type=int, default=16)

    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()
    args.model_name = "D2STGNN"

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
    # logger.info('Adj path: ' + adj_path)
    args.num_feat = args.input_dim
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, "doubletransition")
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]

    dataloader, scaler = load_dataset(data_path, args, logger)
    cl_step = args.cl_epoch * dataloader["train_loader"].num_batch
    warm_step = args.warm_epoch * dataloader["train_loader"].num_batch

    args, engine_template = check_quantile(args, BaseEngine, CQR_Engine)

    model = D2STGNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        # feature=args.feature,
        horizon=args.horizon,
        seq_len=args.seq_len,
        model_args=vars(args),
    )

    loss_fn = "MAE"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5
    )
    # args, engine_template = check_quantile(args, BaseEngine, CQR_Engine)
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
        # cl_step=cl_step,
        # warm_step=warm_step,
        # horizon=args.horizon,
        metric_list=["MAE", "MAPE", "RMSE"],
        args=args,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path, args.export)


if __name__ == "__main__":
    main()
