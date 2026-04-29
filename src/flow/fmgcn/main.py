import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx
try:
    from .fmgcn_model import FMGCN
except ImportError:
    from fmgcn_model import FMGCN


def add_args(parser):
    parser.set_defaults(
        engine_mode="flow_matching",
        fm_flow_weight=0.1,
        fm_context_dim=32,
        fm_hidden_dim=64,
        fm_node_emb_dim=8,
        fm_ode_steps=20,
    )
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--rnn_unit", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    supports = normalize_adj_mx(adj_mx, args.adj_type)
    support_prior = None
    if supports:
        supports = [torch.tensor(s, dtype=torch.float32, device=device) for s in supports]
        support_prior = torch.stack(supports, dim=0).mean(dim=0)
        support_prior = support_prior / support_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return {"support_prior": support_prior}


def build_model(args, node_num, **ctx):
    return FMGCN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim,
        rnn_unit=args.rnn_unit,
        num_layer=args.num_layer,
        cheb_k=args.cheb_k,
        dropout=args.dropout,
        support_prior=ctx.get("support_prior"),
        seq_len=args.seq_len,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="FMGCN",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        init_weights=True,
    )
