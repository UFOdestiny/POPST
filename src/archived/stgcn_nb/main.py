import os
import sys

import numpy as np

sys.path.append(os.path.abspath(__file__ + '/../../../../'))
from stgcn_nb_engine import STGCN_NB_Engine
import torch

torch.set_num_threads(8)

from src.flow.stgcn.stgcn_model import STGCN_NB
from utils.args import get_public_config, get_log_path, print_args
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.graph_algo import normalize_adj_mx
from base.metrics import nb_nll_loss
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    args = parser.parse_args()
    args.model_name = "STGCN_NB"
    if args.quantile:
        args.model_name += "_CQR"
    log_dir = get_log_path(args)

    logger = get_logger(log_dir, __name__, )
    print_args(logger,args) #logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    # device = torch.device(args.device)
    device = torch.device(0)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)


    #logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)

    Ko = args.seq_len - (args.Kt - 1) * 2 * args.block_num
    blocks = []
    blocks.append([args.input_dim])
    for l in range(args.block_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args.horizon])

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STGCN_NB(node_num=node_num,
                     input_dim=args.input_dim,
                     output_dim=args.output_dim,
                     gso=gso,
                     blocks=blocks,
                     Kt=args.Kt,
                     Ks=args.Ks,
                     dropout=args.dropout,
                     feature=args.feature,
                     horizon=args.horizon,
                     )

    # loss_fn = masked_mae
    loss_fn = nb_nll_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = STGCN_NB_Engine(device=device,
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
                             normalize=args.normalize)

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path, args.export)


if __name__ == "__main__":
    main()
