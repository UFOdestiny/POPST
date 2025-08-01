import argparse
import platform
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", type=str, default="")

    parser.add_argument("--dataset", type=str, default="NYISO")  # NYC
    parser.add_argument("--years", type=str, default="2024")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=6)  # flow 12 od 6
    parser.add_argument("--horizon", type=int, default=1)

    parser.add_argument("--feature", type=int, default=1)
    parser.add_argument("--input_dim", type=int, default=1)  # feature
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--normalize", type=bool, default=True)  # Z-Score

    parser.add_argument("--quantile", type=bool, default=True)
    parser.add_argument("--quantile_alpha", type=float, default=0.1)
    parser.add_argument("--hour_day_month", type=bool, default=False)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--export", type=bool, default=True)
    parser.add_argument("--not_print_args", default=False, action="store_true")

    if platform.system().lower() == "linux":
        parser.add_argument(
            "--result_path",
            type=str,
            default="/home/ec2-user/POPST/res/",
        )
    else:
        parser.add_argument(
            "--result_path",
            type=str,
            default="D:/OneDrive - Florida State University/mycode/PopST/res/",
        )

    return parser


def get_log_path(args):
    if platform.system().lower() == "linux":
        log_dir = "/home/ec2-user/POPST/result/{}/{}/".format(
            args.model_name, args.dataset
        )
    else:
        log_dir = (
            r"D:/OneDrive - Florida State University/mycode/PopST/result/{}/{}/".format(
                args.model_name, args.dataset
            )
        )

    # log_dir=""
    return log_dir


def get_data_path():
    if platform.system().lower() == "linux":
        path = "/home/ec2-user/POPST/datasets/"
        # path = '/home/dy23a.fsu/neu24/LargeST-old/data/'
    else:
        path = "D:/OneDrive - Florida State University/mycode/POPST/dataset/"

    # path=""
    return path


def print_args(logger, args):
    if not args.not_print_args:
        for k, v in vars(args).items():
            logger.info("{}: {}".format(k, v))


def check_quantile(args, normal_model, quantile_model):
    if args.quantile:
        assert args.horizon == 1
        assert args.output_dim == 1
        args.horizon = 3
        args.output_dim = 3
        return args, quantile_model
    return args, normal_model
