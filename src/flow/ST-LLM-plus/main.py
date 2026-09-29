import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import torch
from base.runner import run_experiment
from st_llm_plus_model import STLLMPlus
from utils.dataloader import load_adj_from_numpy


def add_args(parser):
    parser.add_argument("--gpt_channel", type=int, default=256, help="conv-side hidden dimension")
    parser.add_argument("--llm_layer", type=int, default=6, help="number of retained GPT2 blocks")
    parser.add_argument("--U", type=int, default=1, help="number of trailing layers fine-tuned with graph attention + LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--pretrained_model", type=str, default="gpt2", help="HuggingFace pretrained model name")
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=5)


def setup(args, data_path, adj_path, node_num, device, logger):
    del args, data_path, node_num, device, logger
    return {"adj_mx": load_adj_from_numpy(adj_path)}


def build_model(args, node_num, **ctx):
    return STLLMPlus(
        adj_mx=ctx["adj_mx"],
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        gpt_channel=args.gpt_channel,
        llm_layer=args.llm_layer,
        U=args.U,
        lora_rank=args.lora_rank,
        dropout=args.dropout,
        pretrained_model=args.pretrained_model,
    )


if __name__ == "__main__":
    run_experiment(
        model_name="ST-LLM-plus",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        make_optimizer=lambda m, a: torch.optim.AdamW(m.parameters(), lr=a.lrate, weight_decay=a.wdecay),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=a.max_epochs, eta_min=1e-6),
    )
