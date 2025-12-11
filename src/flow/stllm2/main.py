import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch

torch.set_num_threads(8)

from base.CQR_engine import CQR_Engine
from stllm_model import STLLM
from base.engine import BaseEngine
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

    # ST-LLM2 模型特定参数
    parser.add_argument("--d_model", type=int, default=96, help="模型维度")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_kv_heads", type=int, default=2, help="KV注意力头数(GQA)")
    parser.add_argument("--d_ff", type=int, default=256, help="前馈网络维度")
    parser.add_argument("--num_layers", type=int, default=3, help="ST-LLM2层数")
    parser.add_argument("--num_experts", type=int, default=4, help="MoE专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="MoE激活的专家数")
    parser.add_argument("--window_size", type=int, default=4, help="滑动窗口大小")

    # 训练参数
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip_grad_value", type=float, default=5)

    args = parser.parse_args()
    args.model_name = "STLLM2"
    if args.quantile:
        args.model_name += "_CQR"
    args.bs = 16

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

    model = STLLM(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        window_size=args.window_size,
        dropout=args.dropout,
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"Total parameters: {total_params:,}")
    # logger.info(f"Trainable parameters: {trainable_params:,}")

    loss_fn = "MAE"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
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
