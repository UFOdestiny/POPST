import os
import glob
import re
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from datetime import datetime


def get_log(folder_path):
    """获取文件夹中最新的日志文件"""
    log_files = glob.glob(os.path.join(folder_path, "*.log"))
    return max(log_files, key=os.path.getmtime) if log_files else None


def read_log(log_path):
    """从日志中读取 Average 行的指标"""
    with open(log_path) as f:
        for line in reversed(f.readlines()):
            if " - Average:" in line:
                return line[31:-1]
    return None


def seconds_between(t1: str, t2: str) -> int:
    """计算两个时间字符串之间的秒数差"""
    fmt = "%Y-%m-%d %H:%M:%S"
    return int(
        abs((datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt)).total_seconds())
    )


def get_time(log_path):
    """获取训练总时间（从 Data shape 到 Average）"""
    with open(log_path) as f:
        t1 = None
        for line in f:
            if " - Log File Path:" in line:
                t1 = line[:19]
            elif " - Average:" in line and t1:
                return seconds_between(t1, line[:19])
    return None


def get_avg_time(log_path):
    """获取每个 epoch 的平均训练时间"""
    with open(log_path) as f:
        epoch, t1 = 1, None
        for line in f:
            if " - Epoch: " in line:
                epoch = int(line.split(",")[0].split()[-1])
            elif " - Data shape:" in line:
                t1 = line[:19]
            elif " - Average:" in line and t1:
                return f"{seconds_between(t1, line[:19]) / epoch:.2f}"
    return None


def get_parameter(log_path):
    """获取模型参数量"""
    with open(log_path) as f:
        for line in f:
            if "Parameters" in line:
                return line.split()[-1]
    return 0


def collect_results(names, datasets, metrics, path):
    """收集所有模型在所有数据集上的结果"""
    rows = []
    for name in names:
        for dataset in datasets:
            log = get_log(f"{path}/{name}/{dataset}")
            if not log:
                continue
            res = read_log(log)
            if res is None:
                continue

            row = {"Dataset": dataset, "Model": name}
            m = {
                k: v for k, v in re.findall(r"(\w+): ([\-\d\.]+)", res) if k in metrics
            }
            row.update(m)
            row.update({"time": get_time(log), "param": get_parameter(log)})
            rows.append(row)
    return rows


def print_df(names, datasets, metrics, path):
    cat_type = CategoricalDtype(categories=names, ordered=True)
    rows = collect_results(names, datasets, metrics, path)
    df = pd.DataFrame(rows)
    df["Model"] = df["Model"].astype(cat_type)
    df_sorted = df.sort_values(by=["Dataset", "Model"])
    print(df_sorted)


if __name__ == "__main__":
    names = [
        # Traditional
        "HL",
        "LSTM",
        "Transformer",
        # ST Graph
        "DCRNN",
        "STGCN",
        "GWNET",
        "DSTAGNN",
        "ASTGCN",
        "AGCRN",
        "STTN",
        "DGCRN",
        # Time Series
        "GluonTS",
        "PatchTST",
        # LLM
        "STLLM",
        "STLLM2",
        # Target
        "Mamba",
        "UMamba",
    ]
    names = [i + "_CQR" for i in names]
    metrics = ["MAE", "MAPE", "RMSE", "MPIW","IS","COV"]
    datasets = ["panhandle"]
    path = "/home/dy23a.fsu/st/result"

    print_df(names, datasets, metrics, path)
