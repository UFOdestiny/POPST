"""Explicit, validated YAML composition with immutable per-run configuration."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import math
import re
import yaml
from config.paths import ROOT


class Config(Mapping):
    def __init__(self, values):
        def freeze(value):
            if isinstance(value, dict):
                return Config(value)
            if isinstance(value, (list, tuple)):
                return tuple(freeze(v) for v in value)
            return value

        object.__setattr__(self, "_values", {key: freeze(value) for key, value in values.items()})

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __setattr__(self, name, value):
        raise TypeError("Configuration is immutable; create a resolved copy")

    def to_dict(self):
        def thaw(value):
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, tuple):
                return [thaw(v) for v in value]
            return value

        return {key: thaw(value) for key, value in self.items()}


def read_yaml(path):
    with Path(path).open() as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return value


def merge(base, extra, strict=False, prefix=""):
    result = deepcopy(base)
    for key, value in extra.items():
        location = f"{prefix}.{key}".strip(".")
        if strict and key not in result:
            raise ValueError(f"Unknown configuration field: {location}")
        old = result.get(key)
        if isinstance(value, dict):
            if old is not None and not isinstance(old, dict):
                raise TypeError(f"{location}: expected {type(old).__name__}")
            result[key] = merge(old or {}, value, strict, prefix=location)
        else:
            if strict and old is not None:
                if isinstance(old, bool):
                    valid = isinstance(value, bool)
                elif isinstance(old, int):
                    valid = isinstance(value, int) and not isinstance(value, bool)
                elif isinstance(old, float):
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                elif isinstance(old, list):
                    valid = isinstance(value, list)
                else:
                    valid = isinstance(value, type(old))
                if not valid:
                    raise TypeError(
                        f"{location}: expected {type(old).__name__}, got {type(value).__name__}"
                    )
            result[key] = value
    return result


def _document(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Config inheritance cycle: {path}")
    value = read_yaml(path)
    parents = value.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    combined = {}
    for parent in parents:
        combined = merge(combined, _document(path.parent / parent, (*stack, path)))
    return merge(combined, value)


def model_registry():
    return read_yaml(ROOT / "config/models/registry.yaml")


def load_config(path=None, overrides=(), *, model=None, dataset=None):
    raw = _document(path) if path else {}
    # Old saved runs carry scheduler metadata; resource allocation now lives in run.sh.
    raw.pop("slurm", None)
    if model:
        raw.setdefault("model", {})["id"] = model
    if dataset:
        raw.setdefault("data", {})["id"] = dataset
    patches = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError("Overrides must be dotted.key=value")
        key, value = item.split("=", 1)
        cursor = patches
        for part in key.split(".")[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[key.split(".")[-1]] = yaml.safe_load(value)
    raw = merge(raw, patches)
    mid = raw.get("model", {}).get("id")
    did = raw.get("data", {}).get("id")
    if mid not in model_registry():
        raise ValueError(f"Unknown model {mid!r}; select model.id from config/models/registry.yaml in a run configuration")
    if not isinstance(did, str) or not re.fullmatch(r"[\w-]+", did):
        raise ValueError("data.id must name a registered dataset")
    dataset_file = ROOT / "config/datasets" / f"{did}.yaml"
    if not dataset_file.exists():
        raise ValueError(f"Unknown dataset {did}")
    base = merge(read_yaml(ROOT / "config/default.yaml"), read_yaml(dataset_file))
    base = merge(base, read_yaml(ROOT / "config/models" / f"{mid}.yaml"))
    component_defaults = {
        "optimizer": {
            "Adam": dict(name="Adam", lr=0.001, weight_decay=0.0, betas=[0.9, 0.999], eps=1e-8),
            "AdamW": dict(name="AdamW", lr=0.001, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8),
            "SGD": dict(name="SGD", lr=0.001, weight_decay=0.0, momentum=0.0, nesterov=False),
            "none": dict(name="none", lr=0.001, weight_decay=0.0),
        },
        "scheduler": {
            "none": dict(name="none"),
            "StepLR": dict(name="StepLR", step_size=200, gamma=0.95),
            "MultiStepLR": dict(name="MultiStepLR", milestones=[50, 100], gamma=0.1),
            "CosineAnnealingLR": dict(
                name="CosineAnnealingLR",
                T_max=raw.get("training", {}).get("max_epochs", base["training"]["max_epochs"]),
                eta_min=0.0,
            ),
        },
    }
    for section, defaults in component_defaults.items():
        current = base["training"][section]
        requested = raw.get("training", {}).get(section, {}).get("name", current["name"])
        if requested not in defaults:
            raise ValueError(f"Unknown {section}: {requested}")
        if current.get("T_max") == "${training.max_epochs}":
            current["T_max"] = defaults.get("CosineAnnealingLR", {}).get("T_max")
        base["training"][section] = merge(
            defaults[requested], current if requested == current["name"] else {}
        )
    value = merge(base, raw, strict=True)
    if value['data']['protocol'] != 'legacy':
        raise ValueError('Only the original legacy data protocol is supported')
    if value["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    if value["training"]["scheduler"].get("T_max") == "${training.max_epochs}":
        value["training"]["scheduler"]["T_max"] = value["training"]["max_epochs"]
    if value["runtime"]["mode"] not in ("train", "test", "calibrate"):
        raise ValueError("runtime.mode must be train, test or calibrate")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", str(value["runtime"]["device"])):
        raise ValueError("runtime.device must be cpu, cuda, or cuda:N")
    for key in ["batch_size", "max_epochs", "patience"]:
        if value["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if not math.isfinite(value["training"]["min_delta"]) or value["training"]["min_delta"] < 0:
        raise ValueError("training.min_delta must be finite and nonnegative")
    for key in ["num_workers", "threads"]:
        if value["runtime"][key] < 0:
            raise ValueError(f"runtime.{key} must be nonnegative")
    for key in ["max_train_samples", "max_val_samples", "max_test_samples"]:
        v = value["runtime"][key]
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
            raise ValueError(f"runtime.{key} must be a positive integer or null")
    if value["calibration"]["mode"] not in ("no", "horizon", "global"):
        raise ValueError("calibration.mode must be no, horizon, or global")
    if value["calibration"]["method"] not in ("split", "aci"):
        raise ValueError("calibration.method must be split or aci")
    if not 0 < value["calibration"]["alpha"] < 1:
        raise ValueError("calibration.alpha must lie between 0 and 1")
    if value["experiment"]["name"] in (".", "..") or not re.fullmatch(
        r"[\w.-]+", value["experiment"]["name"]
    ):
        raise ValueError("experiment.name must be a simple directory name")
    for key in ["seq_len", "horizon", "input_dim", "output_dim", "node_num"]:
        v = value["data"][key]
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
            raise ValueError(f"data.{key} must be a positive integer or null")
    for key in ["directory", "adjacency", "version"]:
        v = value["data"][key]
        if not isinstance(v, str) or Path(v).is_absolute() or ".." in Path(v).parts:
            raise ValueError(f"data.{key} must be a relative path within its configured root")
    if value['data']['version'] != '2025_12to1':
        raise ValueError('Use the original prepared data version 2025_12to1')
    return Config(value)


def resolved_copy(config, changes):
    return Config(merge(config.to_dict(), changes))
