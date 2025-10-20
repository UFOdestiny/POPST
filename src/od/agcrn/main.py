import os
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
sys.path.append("/home/dy23a.fsu/st/")

torch.set_num_threads(8)

from agcrn_model import AGCRN
from agcrn_engine import AGCRN_Engine, AGCRN_Engine_Quantile
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
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--rnn_unit", type=int, default=24)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--clip_grad_value", type=float, default=0)
    args = parser.parse_args()
    args.model_name = "AGCRN_OD"

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

    args, engine_template = check_quantile(args, AGCRN_Engine, AGCRN_Engine_Quantile)

    model = AGCRN(
        node_num=node_num,
        input_dim=node_num,
        output_dim=node_num,
        embed_dim=args.embed_dim,
        rnn_unit=args.rnn_unit,
        num_layer=args.num_layer,
        cheb_k=args.cheb_k,
    )

    loss_fn = "MSE"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    scheduler = None

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
