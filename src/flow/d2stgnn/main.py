import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from d2stgnn_model import D2STGNN
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
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
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    args.num_feat = args.input_dim
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, "doubletransition")
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
    return {}


def build_model(args, node_num, **ctx):
    return D2STGNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        horizon=args.horizon,
        seq_len=args.seq_len,
        model_args=vars(args),
    )


if __name__ == "__main__":
    run_experiment(
        model_name="D2STGNN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        make_optimizer=lambda m, a: torch.optim.Adam(m.parameters(), lr=a.lrate, weight_decay=a.wdecay, eps=1e-8),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5),
    )
