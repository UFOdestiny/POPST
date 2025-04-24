import os
import sys



sys.path.append(os.path.abspath(__file__ + "/../../.."))
sys.path.append("/home/dy23a.fsu/st/")
from base.quantile_engine import Quantile_Engine
import torch
import numpy as np

torch.set_num_threads(8)

from switch_model import SWITCH
from base.engine import BaseEngine
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_adj_from_numpy, load_dataset, get_dataset_info
from utils.log import get_logger
from utils.graph_algo import normalize_adj_mx

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    parser.add_argument(
        "--gcn",
        type=dict,
        default={"in_channel": 6, "out_channel": 256, "hidden_channel": 128},
    )
    parser.add_argument(
        "--tcn",
        type=dict,
        default={"in_channel": 1, "out_channel": 1, "kernel_size": 3, "dilation": 1},
    )
    parser.add_argument("--num_trans", type=int, default=1)
    parser.add_argument("--node_list", type=list, default=[])

    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()

    args.model_name = "SWITCH_OD"
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
    args.node_list = [node_num]

    adj_mx = load_adj_from_numpy(adj_path)
    adj = normalize_adj_mx(adj_mx, 'scalap')[0]
    adj = torch.tensor(adj).to(device)


    dataloader, scaler = load_dataset(data_path, args, logger)
    args, engine_template = check_quantile(args, BaseEngine, Quantile_Engine)

    model = SWITCH(node_num,adj,args)

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
