#!/usr/bin/env python3
"""Result collection and analysis for POPST experiments.

Reads log files from result directories, extracts metrics, efficiency data,
and training statistics, then formats them into comparison tables.

Usage:
    python utils/res.py --path result/Test
    python utils/res.py --log result/Test/STGCN/nyc_mobility/2026-04-16.log
"""

import os
import glob
import re
import argparse
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Log file selection
# ---------------------------------------------------------------------------

def get_log(folder_path, select_metric=None):
    """Get the best or most recent log file from a folder.

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

    best_log, best_value = None, float("inf")
    for log_file in log_files:
        res = read_log(log_file)
        if res is None:
            continue
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


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def read_log(log_path):
    """Read the Average line metrics from a log file."""
    with open(log_path) as f:
        for line in reversed(f.readlines()):
            if " - Average:" in line:
                return line[line.index(" - Average:") + 3:]
    return None


def _parse_efficiency(log_path):
    """Parse the Efficiency section from a log file.

    Returns a dict with keys like 'gpu_peak_mb', 'cpu_rss_mb',
    'inference_ms', 'full_test_s', 'flops', 'total_params', etc.
    """
    result = {}
    in_efficiency = False

    with open(log_path) as f:
        for line in f:
            if "Efficiency" in line and "===" in line:
                in_efficiency = True
                continue
            if in_efficiency and "===" in line and "Efficiency" not in line:
                break
            if not in_efficiency:
                continue

            line = line.strip()
            # Remove timestamp prefix if present
            if " - " in line:
                line = line[line.index(" - ") + 3:].strip()

            if "GPU Peak Allocated" in line:
                result["gpu_peak_mb"] = _extract_float(line)
            elif "GPU Peak Reserved" in line:
                result["gpu_reserved_mb"] = _extract_float(line)
            elif "GPU Current Allocated" in line:
                result["gpu_current_mb"] = _extract_float(line)
            elif "CPU Memory (RSS)" in line:
                result["cpu_rss_mb"] = _extract_float(line)
            elif "Total Parameters" in line:
                result["total_params"] = _extract_int(line)
            elif "Trainable Parameters" in line:
                result["trainable_params"] = _extract_int(line)
            elif "Inference (1 batch)" in line:
                m = re.search(r"([\d.]+)\s*±\s*([\d.]+)\s*ms", line)
                if m:
                    result["inference_ms"] = float(m.group(1))
                    result["inference_std_ms"] = float(m.group(2))
            elif "Full Test Inference" in line:
                m = re.search(r"([\d.]+)\s*s\s*\((\d+)\s*batches\)", line)
                if m:
                    result["full_test_s"] = float(m.group(1))
                    result["test_batches"] = int(m.group(2))
            elif "FLOPs" in line and "N/A" not in line:
                m = re.search(r"([\d.]+)\s*(T|G|M|)FLOPs", line)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2)
                    multiplier = {"T": 1e12, "G": 1e9, "M": 1e6, "": 1}
                    result["flops"] = val * multiplier[unit]
                    result["flops_str"] = f"{m.group(1)} {unit}FLOPs"
            elif "GPU Memory" in line and "GB" in line:
                result["gpu_total_gb"] = _extract_float(line)
            elif re.search(r"^\s*GPU\s*:", line) and "Peak" not in line and "Current" not in line and "Reserved" not in line:
                name = line.split(":")[-1].strip()
                if name and "MB" not in name and "GB" not in name:
                    result["gpu_name"] = name
            elif "CUDA Version" in line:
                result["cuda_version"] = line.split(":")[-1].strip()

    return result


def _extract_float(line):
    """Extract the last float from a line."""
    m = re.findall(r"[\d.]+", line)
    return float(m[-1]) if m else 0.0


def _extract_int(line):
    """Extract integer from a line (handles comma-separated numbers)."""
    m = re.search(r"[\d,]+", line.split(":")[-1])
    return int(m.group().replace(",", "")) if m else 0


def _parse_training_epochs(log_path):
    """Parse all training epoch lines and return a list of dicts.

    Each dict has keys like 'epoch', 'tr_mae', 'v_mae', 'te_mae',
    'tr_time', 'v_time', 'te_time', 'lr', etc.
    """
    epochs = []
    with open(log_path) as f:
        for line in f:
            if " - Epoch: " not in line:
                continue
            # Strip timestamp prefix
            content = line[line.index(" - Epoch:") + 3:]
            m = re.search(r"Epoch:\s*(\d+)", content)
            if not m:
                continue
            epoch_data = {"epoch": int(m.group(1))}

            # Parse all "Key: Value" pairs from content after timestamp
            for key, val in re.findall(r"(\w[\w ]*?):\s*([\d.eE+-]+)", content):
                key_clean = key.strip().lower().replace(" ", "_")
                try:
                    epoch_data[key_clean] = float(val)
                except ValueError:
                    pass

            # Parse timing
            for prefix, skey in [("Tr Time", "tr_time_s"), ("V Time", "v_time_s"), ("Te Time", "te_time_s")]:
                tm = re.search(rf"{prefix}:\s*([\d.]+)\s*s", content)
                if tm:
                    epoch_data[skey] = float(tm.group(1))

            # Parse LR
            lr_m = re.search(r"LR:\s*([\d.eE+-]+)", content)
            if lr_m:
                epoch_data["lr"] = float(lr_m.group(1))

            epochs.append(epoch_data)
    return epochs


def _parse_test_horizons(log_path):
    """Parse per-horizon test results.

    Returns list of dicts: [{'horizon': 1, 'MAE': ..., 'RMSE': ...}, ...]
    """
    horizons = []
    with open(log_path) as f:
        for line in f:
            if "Test Horizon:" not in line:
                continue
            m = re.search(r"Test Horizon:\s*(\d+)", line)
            if not m:
                continue
            h = {"horizon": int(m.group(1))}
            for key, val in re.findall(r"(\w+):\s*([\d.eE+-]+)", line):
                if key != "Horizon":
                    try:
                        h[key] = float(val)
                    except ValueError:
                        pass
            horizons.append(h)
    return horizons


def get_time(log_path):
    """Get total training time (seconds from Log File Path to Average)."""
    with open(log_path) as f:
        t1 = None
        for line in f:
            if " - Log File Path:" in line:
                t1 = line[:19]
            elif " - Average:" in line and t1:
                return seconds_between(t1, line[:19])
    return None


def get_avg_time(log_path):
    """Get average training time per epoch (seconds)."""
    epochs = _parse_training_epochs(log_path)
    if not epochs:
        return None
    total_time = sum(e.get("tr_time_s", 0) for e in epochs)
    return total_time / len(epochs)


def get_parameter(log_path):
    """Get the number of model parameters (in thousands)."""
    eff = _parse_efficiency(log_path)
    if eff.get("total_params"):
        return f"{eff['total_params'] / 1000:.1f}"
    return "0"


def get_memory_stats(log_path):
    """Get GPU and CPU memory usage in MB.

    Returns (gpu_peak_mb, cpu_rss_mb).
    """
    eff = _parse_efficiency(log_path)
    return int(eff.get("gpu_peak_mb", 0)), int(eff.get("cpu_rss_mb", 0))


def seconds_between(t1: str, t2: str) -> int:
    """Calculate the number of seconds between two time strings."""
    fmt = "%Y-%m-%d %H:%M:%S"
    return int(
        abs((datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt)).total_seconds())
    )


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_results(names, datasets, metrics, path, select_metric=None):
    """Collect results for all models across all datasets.

    Returns a list of row dicts with model name, dataset, metrics, and
    efficiency data.
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

            # Test metrics
            m = {
                k: v for k, v in re.findall(r"(\w+): ([\-\d\.]+)", res) if k in metrics
            }
            row.update(m)

            # Efficiency
            eff = _parse_efficiency(log)
            gpu_mem, cpu_mem = get_memory_stats(log)
            row.update({
                "time": get_time(log),
                "avg_time": get_avg_time(log),
                "param_K": get_parameter(log),
                "GPU_MB": gpu_mem,
                "CPU_MB": cpu_mem,
                "inference_ms": eff.get("inference_ms", ""),
                "flops": eff.get("flops_str", ""),
            })
            rows.append(row)
    return rows


def print_df(names, datasets, metrics, path, select_metric=None, title=None):
    """Print a results DataFrame.

    Args:
        names: List of model names
        datasets: List of datasets
        metrics: List of metrics to collect
        path: Results path
        select_metric: Metric for selecting the best log
        title: Optional title for the table
    """
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    cat_type = CategoricalDtype(categories=names, ordered=True)
    rows = collect_results(names, datasets, metrics, path, select_metric)
    if not rows:
        print(f"  No results found in {path}")
        return

    df = pd.DataFrame(rows)
    df["Model"] = df["Model"].astype(cat_type)
    df_sorted = df.sort_values(by=["Dataset", "Model"])
    print(df_sorted.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Standalone log summary
# ---------------------------------------------------------------------------

def summarize_log(log_path):
    """Print a detailed summary of a single log file."""
    print(f"\n{'='*60}")
    print(f"  Log: {os.path.basename(log_path)}")
    print(f"{'='*60}")

    # Test results
    res = read_log(log_path)
    if res:
        print(f"\n  Test Results: {res.strip()}")

    # Efficiency
    eff = _parse_efficiency(log_path)
    if eff:
        print("\n  Efficiency:")
        if "gpu_name" in eff:
            print(f"    GPU              : {eff['gpu_name']}")
        if "total_params" in eff:
            print(f"    Parameters       : {eff['total_params']:,}")
        if "gpu_peak_mb" in eff:
            print(f"    GPU Peak Memory  : {eff['gpu_peak_mb']:.0f} MB")
        if "cpu_rss_mb" in eff:
            print(f"    CPU Memory (RSS) : {eff['cpu_rss_mb']:.0f} MB")
        if "inference_ms" in eff:
            print(f"    Inference (batch): {eff['inference_ms']:.2f} ms")
        if "full_test_s" in eff:
            print(f"    Full Test Time   : {eff['full_test_s']:.3f} s")
        if "flops_str" in eff:
            print(f"    FLOPs            : {eff['flops_str']}")

    # Training summary
    epochs = _parse_training_epochs(log_path)
    if epochs:
        n = len(epochs)
        total = get_time(log_path)
        print(f"\n  Training:")
        print(f"    Epochs           : {n}")
        if total:
            print(f"    Total Time       : {total}s ({total/60:.1f} min)")
        if "tr_mae" in epochs[-1]:
            print(f"    Final Tr MAE     : {epochs[-1]['tr_mae']:.4f}")
        if "v_mae" in epochs[-1]:
            print(f"    Final V MAE      : {epochs[-1]['v_mae']:.4f}")

    print()


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

def _discover_datasets(path):
    """Auto-discover dataset names from a result directory."""
    datasets = set()
    if not os.path.isdir(path):
        return []
    for model_dir in os.listdir(path):
        model_path = os.path.join(path, model_dir)
        if os.path.isdir(model_path):
            for ds in os.listdir(model_path):
                ds_path = os.path.join(model_path, ds)
                if os.path.isdir(ds_path) and glob.glob(os.path.join(ds_path, "*.log")):
                    datasets.add(ds)
    return sorted(datasets)


def _discover_models(path):
    """Auto-discover model names from a result directory."""
    models = []
    if not os.path.isdir(path):
        return []
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            models.append(name)
    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPST result analysis")
    parser.add_argument("--path", type=str, default=None,
                        help="Result path (e.g., result/Test)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Dataset names to include")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Model names to include")
    parser.add_argument("--log", type=str, default=None,
                        help="Summarize a single log file")
    parser.add_argument("--select", type=str, default="MAE",
                        help="Metric for selecting best log (default: MAE)")
    args = parser.parse_args()

    args.path="/home/dy23a.fsu/st/result/NYC_Mobi_15min"

    if args.log:
        summarize_log(args.log)
    elif args.path:
        datasets = args.datasets or _discover_datasets(args.path)
        models = args.models or _discover_models(args.path)
        metrics = ["MAE", "RMSE", "MAPE"]
        if models and datasets:
            print_df(models, datasets, metrics, args.path, args.select,
                     title=f"Results: {args.path}")
        else:
            print(f"No results found in {args.path}")
    else:
        parser.print_help()
