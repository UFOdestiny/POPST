import os
import numpy as np
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine

import torch

torch.set_num_threads(8)

from src.flow.mamba6.mamba_model import UNetMamba
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_dataset, get_dataset_info
from utils.log import get_logger


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
    parser.add_argument("--sample_factor", type=int, default=2)  # 控制多尺度数量
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ffn_expand", type=int, default=2)
    parser.add_argument("--use_multiscale", type=bool, default=True)
    parser.add_argument("--use_temporal_conv", type=bool, default=True)

    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    args = parser.parse_args()

    args.model_name = "Mamba6"
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

    data_path, _, node_num = get_dataset_info(args.dataset)

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
        sample_factor=args.sample_factor,
        dropout=args.dropout,
        ffn_expand=args.ffn_expand,
        use_multiscale=args.use_multiscale,
        use_temporal_conv=args.use_temporal_conv,
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
