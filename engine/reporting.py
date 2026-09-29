"""Collect structured run metrics; never select a run by its test score."""

import csv
import json
from config.paths import load_paths
from config.loader import read_yaml


def report(root=None, output=None):
    paths = load_paths()
    root = paths.resolve(root or paths.results)
    rows = []
    for status_file in sorted(root.rglob("status.json")):
        run = status_file.parent
        status = json.loads(status_file.read_text())
        if not (run / "config.requested.yaml").exists():
            continue
        cfg = read_yaml(run / "config.requested.yaml")
        row = {
            "run": str(run),
            "status": status["status"],
            "model": cfg["model"]["id"],
            "dataset": cfg["data"]["id"],
            "version": cfg["data"]["version"],
            "seed": cfg["training"]["seed"],
        }
        if (run / "metrics.json").exists():
            metrics = json.loads((run / "metrics.json").read_text())
            row.update(metrics.get("average", {}))
            row.update({f"validation_{k}": v for k, v in metrics.get("validation", {}).items()})
        rows.append(row)
    destination = paths.resolve(output or paths.results / "summary.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with destination.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["run", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return destination
