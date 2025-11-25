import os
import numpy as np
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
from base.engine import BaseEngine
from base.quantile_engine import Quantile_Engine

import torch

torch.set_num_threads(8)

from src.flow.umamba.umamba_model import UMamba
from utils.args import (
    get_public_config,
    get_log_path,
    print_args,
    check_quantile,
    get_data_path,
)
from utils.dataloader import load_dataset, get_dataset_info, load_adj_from_numpy
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    """
    获取配置参数的函数。

    主要参数说明：
    - n_mamba_per_block: 每个编码/解码块中的Mamba层数（增加可提升表达能力，但增加计算量）
    - d_model: 隐藏维度（Mamba的工作维度，推荐64-256）
    - num_levels: U-Net层级数（决定最大降采样倍数，推荐3-4）
    - dropout: Dropout比率（防止过拟合，推荐0.1-0.3）
    - step_size: 学习率调度的步长（每多少个epoch降低一次学习率）
    - gamma: 学习率衰减系数（新学习率 = 旧学习率 * gamma）
    - lrate: 初始学习率（推荐1e-3到1e-4）
    - wdecay: L2正则化权重（推荐1e-4到1e-3）

    ========== 图结构参数说明 ==========
    - use_adaptive_adj: 是否使用自适应邻接矩阵学习（从DGCRN灵感）
    - gcn_enabled: 是否启用图卷积层用于空间依赖建模
    - num_heads: 多头图卷积的头数（推荐1-4）
    """
    parser = get_public_config()

    # ========== U-Net Mamba 特定参数 ==========
    parser.add_argument(
        "--n_mamba_per_block",
        type=int,
        default=1,
        help="每个编码/解码块中Mamba层的数量（建议值: 1-3）",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Mamba隐藏维度（建议值: 32-256，根据显存调整）",
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=3,
        help="U-Net层级数，决定最大降采样倍数（建议值: 2-4）",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout比率（建议值: 0.1-0.3）"
    )

    # ========== 优化器参数 ==========
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="学习率调度器的步长（每N个epoch调整一次学习率）",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="学习率衰减系数（建议值: 0.9-0.99）"
    )
    parser.add_argument(
        "--lrate", type=float, default=1e-3, help="初始学习率（建议值: 1e-4到1e-3）"
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=5e-4,
        help="权重衰减/L2正则化（建议值: 1e-5到1e-3）",
    )

    # ========== 图结构参数 ==========
    parser.add_argument(
        "--use_adaptive_adj",
        type=bool,
        default=True,
        help="是否使用自适应邻接矩阵学习（从数据中学习图结构，推荐True）",
    )
    parser.add_argument(
        "--gcn_enabled",
        type=bool,
        default=True,
        help="是否启用图卷积层用于空间依赖建模（推荐True以提升性能）",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="多头图卷积的头数（推荐值: 1-4，增加可提升表达能力）",
    )

    args = parser.parse_args()

    args.model_name = "UMamba"
    log_dir = get_log_path(args)
    logger = get_logger(
        log_dir,
        __name__,
    )
    print_args(logger, args)

    return args, log_dir, logger


def main():
    """
    主训练函数。流程如下：
    1. 配置参数和日志
    2. 设置随机种子保证可复现性
    3. 加载数据集和邻接矩阵
    4. 构建U-Net Mamba模型，融入图结构
    5. 设置优化器和学习率调度
    6. 创建训练引擎并执行训练或评估
    """
    args, log_dir, logger = get_config()

    # ========== 1. 环境初始化 ==========
    set_seed(args.seed)
    device = torch.device(args.device)

    # ========== 2. 加载数据和邻接矩阵 ==========
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    dataloader, scaler = load_dataset(data_path, args, logger)
    args, engine_template = check_quantile(args, BaseEngine, Quantile_Engine)

    # ========== 加载邻接矩阵用于图结构 ==========
    # 邻接矩阵形状: (node_num, node_num)
    # 如果启用图卷积，则传入邻接矩阵；否则为None
    if args.gcn_enabled:
        try:
            predefined_adj = load_adj_from_numpy(adj_path)
            # 转换为torch tensor
            predefined_adj = torch.from_numpy(predefined_adj).float()
        except Exception as e:
            predefined_adj = torch.ones(node_num, node_num)
    else:
        predefined_adj = None

    # ========== 3. 构建改进的U-Net Mamba模型 ==========
    # 融入图结构、自适应邻接矩阵和GCN层
    model = UMamba(
        node_num=node_num,
        input_dim=args.seq_len,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        # 基础参数
        feature=args.feature,
        d_model=args.d_model,
        num_levels=args.num_levels,
        n_mamba_per_block=args.n_mamba_per_block,
        dropout=args.dropout,
        # 图结构参数（改进关键）
        predefined_adj=predefined_adj,
        use_adaptive_adj=args.use_adaptive_adj,
        gcn_enabled=args.gcn_enabled,
        num_heads=args.num_heads,
    )

    # ========== 4. 损失函数和优化器 ==========
    loss_fn = "MAE"

    # Adam优化器：学习率为args.lrate，权重衰减为args.wdecay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lrate,
        weight_decay=args.wdecay,
    )

    # 阶梯式学习率调度：每step_size个epoch将学习率乘以gamma
    # 这有助于在训练后期稳定收敛
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # ========== 5. 构建训练引擎 ==========
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
        clip_grad_value=0,  # 梯度裁剪（0表示不使用）
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

    # ========== 6. 开始训练或评估 ==========
    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path, args.export)


if __name__ == "__main__":
    main()
