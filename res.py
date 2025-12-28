#!/blue/gtyson.fsu/dy23a.fsu/conda/envs/st/bin/python3
import os
import glob
import re
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from datetime import datetime


def get_log(folder_path, select_metric=None):
    """获取文件夹中的日志文件

    Args:
        folder_path: 日志文件夹路径
        select_metric: 如果指定，则返回该指标数值最低的日志文件；否则返回最新的日志文件
    """
    log_files = glob.glob(os.path.join(folder_path, "*.log"))
    if not log_files:
        return None

    if select_metric is None:
        return max(log_files, key=os.path.getmtime)

    # 根据指定 metric 选择最优的 log
    best_log = None
    best_value = float("inf")

    for log_file in log_files:
        res = read_log(log_file)
        if res is None:
            continue
        # 解析指标
        m = dict(re.findall(r"(\w+): ([\-\d\.]+)", res))
        if select_metric in m:
            try:
                value = float(m[select_metric])
                if value < best_value:
                    best_value = value
                    best_log = log_file
            except ValueError:
                continue

    return best_log if best_log else max(log_files, key=os.path.getmtime)


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


def collect_results(names, datasets, metrics, path, select_metric=None):
    """收集所有模型在所有数据集上的结果

    Args:
        names: 模型名称列表
        datasets: 数据集列表
        metrics: 要收集的指标列表
        path: 结果路径
        select_metric: 如果指定，则选择该指标数值最低的日志文件
    """
    rows = []
    for name in names:
        for dataset in datasets:
            log = get_log(f"{path}/{name}/{dataset}", select_metric)
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


def print_df(names, datasets, metrics, path, select_metric=None):
    """打印结果 DataFrame

    Args:
        names: 模型名称列表
        datasets: 数据集列表
        metrics: 要收集的指标列表
        path: 结果路径
        select_metric: 如果指定，则选择该指标数值最低的日志文件
    """
    cat_type = CategoricalDtype(categories=names, ordered=True)
    rows = collect_results(names, datasets, metrics, path, select_metric)
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
        "Mamba2",
        "Mamba3",
        "Mamba4",
        "Mamba5",
        "Mamba6",
        "Mamba7",
    ]

    names = [i + "_CQR" for i in names]

    metrics = ["MAE", "RMSE", "MPIW", "IS", "COV"]

    # 指定选择最优 log 的指标（选择该指标数值最低的 log），设为 None 则使用最新的 log
    select_metric = "MAE"  # 可选: "MAE", "RMSE", "MPIW", "IS", "COV" 等

    datasets = ["safegraph_fl"]
    path = "/home/dy23a.fsu/st/result/fl"
    print_df(names, datasets, metrics, path, select_metric)
    path = "/home/dy23a.fsu/st/result/mfl"
    print_df(names, datasets, metrics, path, select_metric)

    datasets = ["safegraph_ny"]
    path = "/home/dy23a.fsu/st/result/ny"
    print_df(names, datasets, metrics, path, select_metric)
    path = "/home/dy23a.fsu/st/result/mny"
    print_df(names, datasets, metrics, path, select_metric)

    datasets = ["safegraph_ca"]
    path = "/home/dy23a.fsu/st/result/ca"
    print_df(names, datasets, metrics, path, select_metric)
    path = "/home/dy23a.fsu/st/result/mca"
    print_df(names, datasets, metrics, path, select_metric)

    datasets = ["safegraph_tx"]
    path = "/home/dy23a.fsu/st/result/tx"
    print_df(names, datasets, metrics, path, select_metric)
    path = "/home/dy23a.fsu/st/result/mtx"
    print_df(names, datasets, metrics, path, select_metric)
