import os
import sys
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
import torch

from trustenergy_model import TrustE
from trustenergy_engine import TrustE_Engine
from base.metrics import masked_mae

torch.set_num_threads(3)
from utils.graph_algo import normalize_adj_mx
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.log import get_logger
from base.engine import BaseEngine
from base.quantile_engine import Quantile_Engine


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    parser.add_argument("--rank_s", type=int, default=512)
    parser.add_argument("--rank_t", type=int, default=512)
    parser.add_argument("--hidden_dim_s", type=int, default=64)
    parser.add_argument("--hidden_dim_t", type=int, default=64)

    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip_grad_value", type=float, default=0)

    args = parser.parse_args()
    args.model_name = "TrustEnergy_TEST"
    args.dataset="NYISO_HDM"
    args.hour_day_month=True

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
    device = torch.device(0)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    # gso = torch.tensor(gso).to(device)

    dataloader, scaler = load_dataset(data_path, args, logger)

    args, engine_template = check_quantile(args, BaseEngine, TrustE_Engine)

    model = TrustE(
        A=gso,
        seq_len=args.seq_len,
        node_num=node_num,
        hidden_dim_t=args.hidden_dim_t,
        hidden_dim_s=args.hidden_dim_s,
        rank_t=args.rank_t,
        rank_s=args.rank_s,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.horizon,
        device=device,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        
    )

    loss_fn = "MAE"
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
        # hour_day_month=args.hour_day_month,
        metric_list=["MAE", "MAPE", "RMSE"],
        args=args,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path)


if __name__ == "__main__":
    main()
