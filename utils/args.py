import argparse
import os
import platform
import re

import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ENGINE_MODES = ("standard", "flow_matching")


def get_public_config():
    """Create argument parser with common training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", type=str, default="")

    parser.add_argument("--dataset", type=str, default="CAISO")
    parser.add_argument("--years", type=str, default="2018")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)

    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=None)

    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")

    parser.add_argument("--quantile", action="store_true", default=False)
    parser.add_argument("--quantile_alpha", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument("--not_print_args", action="store_true", default=False)
    parser.add_argument("--proj", type=str, default="")
    parser.add_argument(
        "--engine_mode",
        type=str,
        default="standard",
        choices=ENGINE_MODES,
    )
    parser.add_argument("--fm_flow_weight", type=float, default=0.1)
    parser.add_argument("--fm_ode_steps", type=int, default=20)
    parser.add_argument("--fm_num_samples", type=int, default=10)
    parser.add_argument("--fm_context_dim", type=int, default=32)
    parser.add_argument("--fm_time_dim", type=int, default=32)
    parser.add_argument("--fm_hidden_dim", type=int, default=64)
    parser.add_argument("--fm_node_emb_dim", type=int, default=8)
    parser.add_argument(
        "--fm_output_activation",
        type=str,
        default="relu",
        choices=["none", "relu"],
    )

    return parser


# System path configuration
_SYSTEM_CONFIG = {
    "linux": {
        "log_base": "/home/dy23a.fsu/st/result",
        "data_base": "/blue/gtyson.fsu/dy23a.fsu/datasets",
    },
    "windows": {
        "log_base": r"E:/OneDrive - Florida State University/mycode/PopST/result",
        "data_base": r"E:/OneDrive - Florida State University/mycode/POPST/dataset",
    },
}


def _get_system_type():
    return "linux" if platform.system().lower() == "linux" else "windows"


def get_log_path(args):
    sys_type = _get_system_type()
    base_path = _SYSTEM_CONFIG[sys_type]["log_base"]

    if sys_type == "linux":
        return f"{base_path}/{args.proj}/{args.model_name}/{args.dataset}/"
    return f"{base_path}/{args.model_name}/{args.dataset}/"


def get_data_path():
    sys_type = _get_system_type()
    return _SYSTEM_CONFIG[sys_type]["data_base"] + "/"


def _fmt_value(v):
    """Format a value for display, showing shapes for tensors/arrays."""
    if isinstance(v, torch.Tensor):
        return f"Tensor{list(v.shape)}"
    if isinstance(v, np.ndarray):
        return f"ndarray{list(v.shape)}"
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (torch.Tensor, np.ndarray)):
        shapes = [f"{type(x).__name__}{list(x.shape)}" for x in v]
        return f"[{', '.join(shapes)}]"
    return v


def print_args(logger, args):
    if args.not_print_args:
        return

    groups = {
        "Data": ["dataset", "years", "seq_len", "horizon", "input_dim", "output_dim", "normalize"],
        "Model": ["model_name"],
        "Training": ["bs", "max_epochs", "patience", "lrate", "wdecay", "clip_grad_norm", "seed"],
        "Engine": [
            "engine_mode",
            "fm_flow_weight",
            "fm_ode_steps",
            "fm_num_samples",
            "fm_context_dim",
            "fm_time_dim",
            "fm_hidden_dim",
            "fm_node_emb_dim",
            "fm_output_activation",
        ],
        "Quantile": ["quantile", "quantile_alpha"],
        "System": ["device", "mode", "model_path", "export", "proj", "comment"],
    }

    known = {k for keys in groups.values() for k in keys}
    all_vars = vars(args)

    for group_name, keys in groups.items():
        present = [(k, all_vars[k]) for k in keys if k in all_vars]
        if not present:
            continue
        logger.info(f"--- {group_name} ---")
        for k, v in present:
            logger.info(f"  {k:20s}: {_fmt_value(v)}")

    extra = {k: v for k, v in all_vars.items() if k not in known and k != "not_print_args"}
    if extra:
        logger.info("--- Model-specific ---")
        for k, v in extra.items():
            logger.info(f"  {k:20s}: {_fmt_value(v)}")


def validate_shared_args(args):
    if args.quantile and args.engine_mode != "standard":
        raise ValueError("Quantile mode only supports --engine_mode standard right now.")
    if args.fm_ode_steps < 1:
        raise ValueError("--fm_ode_steps must be >= 1.")
    if args.fm_num_samples < 1:
        raise ValueError("--fm_num_samples must be >= 1.")
    if args.fm_context_dim < 1:
        raise ValueError("--fm_context_dim must be >= 1.")
    if args.fm_time_dim < 1:
        raise ValueError("--fm_time_dim must be >= 1.")
    if args.fm_hidden_dim < 1:
        raise ValueError("--fm_hidden_dim must be >= 1.")
    if args.fm_node_emb_dim < 0:
        raise ValueError("--fm_node_emb_dim must be >= 0.")
    return args


def resolve_engine_template(args, standard_engine, quantile_engine, flow_matching_engine):
    if args.quantile:
        return quantile_engine
    if args.engine_mode == "flow_matching":
        return flow_matching_engine
    return standard_engine


def get_fm_wrapper_kwargs(args):
    return {
        "context_dim": args.fm_context_dim,
        "time_dim": args.fm_time_dim,
        "flow_hidden_dim": args.fm_hidden_dim,
        "node_emb_dim": args.fm_node_emb_dim,
    }


def set_seed(seed):
    """Set random seed for reproducibility across numpy, torch CPU and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def tuple_type(strings):
    """Convert string to integer tuple. Supports formats: '1,2,3' or '(1,2,3)'."""
    cleaned = re.sub(r"[()\\s]", "", strings)
    try:
        return tuple(map(int, cleaned.split(",")))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {strings}") from e
