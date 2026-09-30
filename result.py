"""Print all saved experiment results, grouped by dataset, using pandas.

Usage: python result.py [--results-dir results] [--dataset DATASET] [--precision 6]
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


METRIC_COLUMNS = [
    "MAE", "MAPE", "MSE", "RMSE", "F1", "TZR", "KL", "CRPS",
    "MPIW", "IS", "COV", "WINK", "NLL", "MGAU", "Quantile",
]


def read_mapping(path):
    """Allow incomplete artifacts without losing the rest of a run's results."""
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        value = yaml.safe_load(content) if path.suffix == ".yaml" else json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("expected a mapping")
        return value
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Warning: {path}: {exc}", file=sys.stderr)
        return {}


def flatten(mapping, prefix=""):
    """Preserve additional test metric names and nested results."""
    values = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            values.update(flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            values[name] = json.dumps(value, ensure_ascii=False)
        else:
            values[name] = value
    return values


def collect_results(root):
    """Use the latest completed test result for each model/seed/version."""
    run_dirs = sorted({
        path.parent
        for name in ("config.requested.yaml", "config.resolved.yaml", "status.json", "metrics.json")
        for path in root.rglob(name)
    })
    rows = []
    for run in run_dirs:
        cfg = read_mapping(run / "config.resolved.yaml")
        if not cfg:
            cfg = read_mapping(run / "config.requested.yaml")
        status = read_mapping(run / "status.json")
        metrics = read_mapping(run / "metrics.json")
        row = {
            "dataset": cfg.get("data", {}).get("id", "unknown"),
            "model": cfg.get("model", {}).get("id", "unknown"),
            "version": cfg.get("data", {}).get("version"),
            "seed": cfg.get("training", {}).get("seed"),
            "status": status.get("status", "unknown"),
            "_run_time": run.name,
        }
        # Training with evaluate_test=false writes validation results to metrics.json.
        is_test = not (
            cfg.get("runtime", {}).get("mode", "train") == "train"
            and cfg.get("training", {}).get("evaluate_test") is False
        )
        test_metrics = metrics.get("average", {}) if is_test else {}
        row["_has_test"] = bool(test_metrics) and row["status"] == "succeeded"
        if row["_has_test"]:
            row.update(flatten(test_metrics))
        rows.append(row)
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    # Prefer an existing completed result over a newer failed/running retry.
    results = results.sort_values(["_has_test", "_run_time"]).drop_duplicates(
        ["dataset", "model", "version", "seed"], keep="last",
    )
    results = results.drop(columns=["_has_test", "_run_time"])
    for metric in METRIC_COLUMNS:
        if metric not in results:
            results[metric] = float("nan")
    return results


def print_results(results, precision=6):
    if results.empty:
        print("No experiment results found.")
        return
    float_format = lambda value: f"{value:.{precision}f}"  # noqa: E731
    print(f"Results: {len(results)} | Datasets: {results['dataset'].nunique()}")
    for dataset, group in results.groupby("dataset", sort=True, dropna=False):
        print(f"\n{'=' * 80}\nDataset: {dataset}")
        identifiers = ["model"]
        for column in ("version", "seed"):
            if group[column].nunique(dropna=False) > 1:
                identifiers.append(column)
        extras = [
            column for column in group
            if column not in ["dataset", "model", "version", "seed", "status", *METRIC_COLUMNS]
            and group[column].notna().any()
        ]
        table = group.sort_values(["model", "seed"])[identifiers + METRIC_COLUMNS + extras]
        print(table.to_string(index=False, float_format=float_format, na_rep=""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results",
        help="Results directory (default: results next to this script)",
    )
    parser.add_argument("--dataset", help="Only print this exact dataset ID")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places (default: 6)")
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        parser.error(f"Results directory does not exist: {args.results_dir}")
    if args.precision < 0:
        parser.error("--precision must be non-negative")
    results = collect_results(args.results_dir)
    if args.dataset and not results.empty:
        results = results.loc[results["dataset"] == args.dataset]
    print_results(results, args.precision)


if __name__ == "__main__":
    main()
