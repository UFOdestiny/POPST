#!/usr/bin/env python3
import os
import glob
import re
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from datetime import datetime


def get_log(folder_path, select_metric=None):
    """Get log files from a folder.

    Args:
        folder_path: Path to the log folder
        select_metric: If specified, return the log file with the lowest value
                       for this metric; otherwise return the most recent log file
    """
    log_files = glob.glob(os.path.join(folder_path, "*.log"))
    if not log_files:
        return None

    if select_metric is None:
        return max(log_files, key=os.path.getmtime)

    # Select the best log based on the specified metric
    best_log = None
    best_value = float("inf")

    for log_file in log_files:
        res = read_log(log_file)
        if res is None:
            continue
        # Parse metrics
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
    """Read the Average line metrics from a log file."""
    with open(log_path) as f:
        for line in reversed(f.readlines()):
            if " - Average:" in line:
                return line[31:-1]
    return None


def seconds_between(t1: str, t2: str) -> int:
    """Calculate the number of seconds between two time strings."""
    fmt = "%Y-%m-%d %H:%M:%S"
    return int(
        abs((datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt)).total_seconds())
    )


def get_time(log_path):
    """Get total training time (from Log File Path to Average)."""
    with open(log_path) as f:
        t1 = None
        for line in f:
            if " - Log File Path:" in line:
                t1 = line[:19]
            elif " - Average:" in line and t1:
                return seconds_between(t1, line[:19])
    return None


def get_avg_time(log_path):
    """Get average training time per epoch."""
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
    """Get the number of model parameters."""
    with open(log_path) as f:
        for line in f:
            if "Parameters" in line:
                return str(int(line.split()[-1]) / 1000)
    return 0


def get_memory_stats(log_path):
    """Get Max Peak GPU Memory and Max Peak CPU Memory."""
    gpu_mem, cpu_mem = 0.0, 0.0
    with open(log_path) as f:
        for line in f:
            if "Max Peak GPU Memory:" in line:
                try:
                    gpu_mem = float(line.split(":")[-1].replace("MB", "").strip())
                except ValueError:
                    pass
            elif "Max Peak CPU Memory:" in line:
                try:
                    cpu_mem = float(line.split(":")[-1].replace("MB", "").strip())
                except ValueError:
                    pass
    return int(gpu_mem), int(cpu_mem)


def collect_results(names, datasets, metrics, path, select_metric=None):
    """Collect results for all models across all datasets.

    Args:
        names: List of model names
        datasets: List of datasets
        metrics: List of metrics to collect
        path: Results path
        select_metric: If specified, select the log file with the lowest value for this metric
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

            gpu_mem, cpu_mem = get_memory_stats(log)
            row.update(
                {
                    "time": get_time(log),
                    "param": get_parameter(log),
                    "GPU_MB": gpu_mem,
                    "CPU_MB": cpu_mem,
                }
            )
            rows.append(row)
    return rows


def print_df(names, datasets, metrics, path, select_metric=None):
    """Print a results DataFrame.

    Args:
        names: List of model names
        datasets: List of datasets
        metrics: List of metrics to collect
        path: Results path
        select_metric: If specified, select the log file with the lowest value for this metric
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

    names_nocqr = [i for i in names]
    names = [i + "_CQR" for i in names]

    metrics = ["MAE", "RMSE", "MAPE", "MPIW", "IS", "COV"]

    # Specify the metric for selecting the best log (picks the log with the
    # lowest value for this metric). Set to None to use the most recent log.
    select_metric = "MAE"  # Options: "MAE", "RMSE", "MPIW", "IS", "COV", etc.

    datasets = ["Tallahassee"]
    path = "/home/dy23a.fsu/st/result/FL1"
    print_df(names, datasets, metrics, path, select_metric)

    datasets = ["Tallahassee"]
    path = "/home/dy23a.fsu/st/result/FL2"
    print_df(names, datasets, metrics, path, select_metric)

    datasets = ["CAISO"]
    path = "/home/dy23a.fsu/st/result/CA"
    print_df(names, datasets, metrics, path, select_metric)

    datasets = ["NYISO"]
    path = "/home/dy23a.fsu/st/result/NY"
    print_df(names, datasets, metrics, path, select_metric)
