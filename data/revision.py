"""Leakage-free OD datasets, built locally from the shared count arrays."""

import json
import shutil
from pathlib import Path

import numpy as np

from config.loader import load_config
from config.paths import load_paths
from data.preprocessing import MinMaxScaler, _scaler_to_meta, reconstruct_scaler


DATASETS = {
    "dc_bike": "dc_od_60min_bike",
    "nyc_fhv": "nyc_manhattan_od_15min_fhv",
    "chicago_taxi": "chicago_od_15min_taxi",
    "nyc_taxi": "nyc_manhattan_od_15min_taxi",
    "chicago_tnp": "chicago_od_15min_tnp",
    "chicago_bike": "chicago_od_15min_bike",
}
SPLITS = ("train", "val", "tune", "fit", "cal", "test")
FRACTIONS = (0, .80, .84, .86, .88, .90, 1.)


def split_indices(length, history, horizon):
    """Assign by target timestamp; purge origins whose H targets cross a boundary."""
    boundaries = [int(length * f) for f in FRACTIONS]
    return {
        name: np.arange(max(history - 1, start - 1), end - horizon, dtype=np.int64)
        for name, start, end in zip(SPLITS, boundaries, boundaries[1:])
    }, boundaries


def _counts(source, info, paths, raw_root):
    old_path = Path(info["config"]["data_path"])
    candidates = list(raw_root.rglob(old_path.name))
    if len(candidates) == 1:
        raw = np.load(candidates[0], mmap_mode="r")
        fmt = info["config"]["fmt"]
        if "D" not in fmt:
            raw = raw[..., None]
            fmt += "D"
        pos = fmt.index("T")
        raw = raw.transpose([pos] + [i for i in range(raw.ndim) if i != pos])
        if tuple(raw.shape) != tuple(info["raw_data"]["shape"]):
            raise ValueError("Raw data shape differs from the supplied dataset")
        if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw != np.rint(raw)):
            raise ValueError("Expected finite nonnegative integer counts")
        return raw, {"kind": "raw_counts", "file": str(candidates[0].relative_to(raw_root))}
    # Recover count data only if the inverse is unambiguously integral.
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    with np.load(source / "his.npz") as archive:
        raw = reconstruct_scaler(meta).inverse_transform(archive["data"], device="cpu")
    error = float(np.max(np.abs(raw - np.rint(raw))))
    if not np.isfinite(raw).all() or np.any(raw < -1e-5) or error > .001:
        raise ValueError("Cannot recover counts accurately; original raw counts are required")
    return np.rint(raw).astype(np.float32), {
        "kind": "inverse_then_verified_integer_counts", "max_roundtrip_error": error,
        "file": str(source.relative_to(paths.datasets)),
    }


def prepare_dataset(key, *, raw_root, output_root, horizon=1, history=12, paths=None):
    paths = paths or load_paths()
    cfg = load_config(model="od/pdr_reg", dataset=DATASETS[key])
    version = f"revision_{history}to{horizon}"
    folder = Path(output_root) / cfg.data.directory / version
    if (folder / "audit.json").exists():
        audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        if audit["protocol"] != "revision-1" or audit["horizon"] != horizon:
            raise ValueError("Incompatible prepared data; use a new protocol version")
        return folder
    source = paths.datasets / cfg.data.directory / cfg.data.version
    info = json.loads((source / "info.json").read_text(encoding="utf-8"))
    raw, provenance = _counts(source, info, paths, Path(raw_root))
    indices, boundaries = split_indices(len(raw), history, horizon)
    train_end = boundaries[1]
    flat = raw[:train_end].reshape(-1, raw.shape[-1])
    scaler = MinMaxScaler(
        data_min=np.log1p(flat.min(axis=0)), data_max=np.log1p(flat.max(axis=0)), use_log1p=True
    )
    if np.any(scaler.data_max_.numpy() == scaler.data_min_.numpy()):
        raise ValueError("Constant training channel requires an explicit scaling policy")
    folder.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(folder / "his.npy", mode="w+", dtype="float32", shape=raw.shape)
    for start in range(0, len(raw), 256):
        target[start:start + 256] = scaler.transform(raw[start:start + 256])
    target.flush()
    del target
    statistics = {}
    for name, start, end in zip(SPLITS, boundaries, boundaries[1:]):
        block = raw[start:end]
        positive = int(np.count_nonzero(block))
        idx = indices[name]
        if not len(idx):
            raise ValueError(f"Empty {name} for horizon {horizon}")
        np.save(folder / f"idx_{name}.npy", idx)
        # Targets are identical across methods, including the trivial baselines.
        statistics[name] = {
            "time_start": start, "time_end_exclusive": end,
            "cells": int(block.size), "nonzero_cells": positive,
            "zero_fraction": 1 - positive / block.size,
            "mean_count": float(block.mean()), "max_count": float(block.max()),
            "forecast_origins": len(idx), "first_target": int(idx[0] + 1),
            "last_target": int(idx[-1] + horizon),
        }
    adjacency = Path(output_root) / cfg.data.adjacency
    adjacency.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.datasets / cfg.data.adjacency, adjacency)
    meta = {**_scaler_to_meta(scaler), "data_shape": list(raw.shape),
            "splits": {k: len(v) for k, v in indices.items()}, "scaler_fit_end_exclusive": train_end}
    new_info = {"config": {"seq_length_x": history, "seq_length_y": horizon},
                "raw_data": {"shape": list(raw.shape)}, "protocol": "revision-1"}
    audit = {
        "protocol": "revision-1", "dataset": DATASETS[key], "history": history,
        "horizon": horizon, "frequency": cfg.data.frequency, "source": provenance,
        "scaler_fit": [0, train_end], "splits": statistics,
        "split_fractions": list(FRACTIONS), "target_overlap": False,
    }
    for name, obj in [("meta", meta), ("info", new_info), ("audit", audit)]:
        (folder / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return folder
