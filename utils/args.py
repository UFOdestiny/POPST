import argparse
import os
import platform
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", type=str, default="")

    parser.add_argument("--dataset", type=str, default="panhandle")  # NYC
    parser.add_argument("--years", type=str, default="2018")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=3)

    parser.add_argument("--feature", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=4)  # feature
    parser.add_argument("--output_dim", type=int, default=4)

    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")

    parser.add_argument("--quantile", action="store_true", default=False)
    parser.add_argument("--quantile_alpha", type=float, default=0.1)
    parser.add_argument("--hour_day_month", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument("--not_print_args", action="store_true", default=False)
    parser.add_argument("--proj", type=str, default="")

    return parser


# 系统路径配置
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


def print_args(logger, args):
    if not args.not_print_args:
        for k, v in vars(args).items():
            logger.info(f"{k:20s}: {v}")


def check_quantile(args, normal_model, quantile_model):
    if args.quantile:
        return args, quantile_model
    return args, normal_model


def tuple_type(strings):
    """将字符串转换为整数元组。支持格式: '1,2,3' 或 '(1,2,3)'"""
    # 移除括号和空格
    cleaned = re.sub(r"[()\\s]", "", strings)
    try:
        return tuple(map(int, cleaned.split(",")))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {strings}") from e
