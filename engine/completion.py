"""Find successful runs by comparing saved configuration values directly."""

from copy import deepcopy
import json
from pathlib import Path

import yaml


def _comparison_config(values):
    values = deepcopy(values)
    values.pop("experiment", None)
    values.pop("slurm", None)
    runtime = values.get("runtime", {})
    for key in ("skip_completed", "device", "quiet", "threads", "num_workers", "pin_memory"):
        runtime.pop(key, None)
    return values


def find_completed(config, paths):
    """Return (directory, checkpoint, metrics), or None for unfinished/different runs."""
    if not config.runtime.skip_completed:
        return None
    requested = _comparison_config(config.to_dict())
    for status_path in sorted(paths.results.glob("*/*/status.json"), reverse=True):
        directory = status_path.parent
        try:
            status = json.loads(status_path.read_text())
            if not isinstance(status, dict) or status.get("status") != "succeeded":
                continue
            saved = yaml.safe_load((directory / "config.requested.yaml").read_text())
            if not isinstance(saved, dict) or _comparison_config(saved) != requested:
                continue
            metrics = json.loads((directory / "metrics.json").read_text())
            if not isinstance(metrics, dict) or not metrics:
                continue
            checkpoint = Path(status["checkpoint"]) if status.get("checkpoint") else None
            if checkpoint is not None and not checkpoint.is_file():
                continue
            if config.runtime.profile and not (directory / "efficiency.json").is_file():
                continue
            return directory, checkpoint, metrics
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
            # Interrupted writes and incomplete/corrupt artifacts are never reused.
            continue
    return None
