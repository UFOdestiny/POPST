import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch
from base.runner import run_experiment
from src.flow.mamba7.mamba_model import UNetMamba
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    parser.add_argument("--num_layers", type=int, default=3, help="ST-Mamba块数量")
    parser.add_argument("--d_model", type=int, default=96, help="模型维度")
    parser.add_argument("--d_ff", type=int, default=256, help="FFN/MoE中间维度")
    parser.add_argument("--graph_embed_dim", type=int, default=16, help="图嵌入维度")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout率")
    parser.add_argument("--num_experts", type=int, default=4, help="MoE专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="MoE激活的专家数")
    parser.add_argument("--num_graph_layers", type=int, default=1)
    parser.add_argument("--gate_init", type=float, default=-2.0)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "scalap")[0]
    gso = torch.tensor(gso).to(device)
    return {"adj": gso}


def build_model(args, node_num, **ctx):
    return UNetMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        num_layers=args.num_layers,
        d_model=args.d_model,
        feature=args.input_dim,
        adj=ctx["adj"],
        graph_embed_dim=args.graph_embed_dim,
        dropout=args.dropout,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="Mamba7",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        make_optimizer=lambda m, a: torch.optim.AdamW(m.parameters(), lr=a.lrate, weight_decay=a.wdecay),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.max_epochs, eta_min=1e-6),
    )
