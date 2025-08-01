import os
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + "/../../../../"))
from base.quantile_engine import Quantile_Engine

import torch

torch.set_num_threads(8)

from stgode_model import STGODE
from base.engine import BaseEngine
from utils.args import get_public_config, get_log_path, print_args, check_quantile
from utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from utils.log import get_logger
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--tpd", type=int, default=96, help="time per day")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--thres", type=float, default=0.6)

    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--clip_grad_value", type=float, default=0)
    args = parser.parse_args()

    args.model_name = "STGODE_OD"
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

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    sp_matrix = adj_mx + np.transpose(adj_mx)
    sp_matrix = normalize_adj_mx(sp_matrix).to(device)

    se_matrix = construct_se_matrix(data_path, args)
    se_matrix = normalize_adj_mx(se_matrix).to(device)

    dataloader, scaler = load_dataset(data_path, args, logger)

    args, engine_template = check_quantile(args, BaseEngine, Quantile_Engine)

    model = STGODE(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        A_sp=sp_matrix,
        A_se=se_matrix,
        horizon=args.horizon,
    )

    loss_fn = "MSE"  # masked_mae
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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


def construct_se_matrix(data_path, args):
    # def compute_and_fill(i, j):
    #     dist = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
    #     print(i,j)
    #     return i, j, dist

    ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
    data = ptr["data"][..., 0]
    sample_num, node_num = data.shape

    data_mean = np.mean(
        [
            data[args.tpd * i : args.tpd * (i + 1)]
            for i in range(sample_num // args.tpd)
        ],
        axis=0,
    )
    data_mean = data_mean.T

    dist_matrix = np.zeros((node_num, node_num))

    # pairs = [(i, j) for i in range(node_num) for j in range(i, node_num)]
    # results = Parallel(n_jobs=-1)(delayed(compute_and_fill)(i, j) for i, j in pairs)
    # dist_matrix = np.zeros((node_num, node_num))
    # for i, j, dist in results:
    #     dist_matrix[i][j] = dist
    #     dist_matrix[j][i] = dist

    # print("DONE!!!!!!!!!!!!!!!")

    for i in range(node_num):
        for j in range(i, node_num):
            dist = euclidean(data_mean[i], data_mean[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

            # print(i,j)

    # for i in range(node_num):
    #     for j in range(i, node_num):
    #         print(i,j)
    #         dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    # for i in range(node_num):
    #     for j in range(i):
    #         dist_matrix[i][j] = dist_matrix[j][i]

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    dist_matrix = np.exp(-(dist_matrix**2) / args.sigma**2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres] = 1
    return dtw_matrix


def normalize_adj_mx(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(
        np.multiply(diag.reshape((-1, 1)), adj_mx), diag.reshape((1, -1))
    )
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


if __name__ == "__main__":
    main()
