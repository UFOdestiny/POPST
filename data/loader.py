"""Shared, validated time-series windows for node flow and OD data."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config.loader import resolved_copy
from data.preprocessing import reconstruct_scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, data, indices, seq_len, horizon):
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices)
        self.x_offsets = np.arange(-(seq_len - 1), 1)
        self.y_offsets = np.arange(1, horizon + 1)
        if not len(self.indices):
            raise ValueError("Empty split")
        if self.indices.min() + self.x_offsets.min() < 0 or self.indices.max() + horizon >= len(
            data
        ):
            raise ValueError("Window exceeds dataset bounds; regenerate indices for this window")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        return torch.from_numpy(self.data[t + self.x_offsets]), torch.from_numpy(
            self.data[t + self.y_offsets]
        )


class LoaderAdapter:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        logger=None,
        name=None,
    ):
        self.dataset = dataset
        self.bs = batch_size
        self.size = len(dataset)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.num_batch = len(self.loader)
        if logger:
            logger.info(f"{name}: {self.size} samples, {self.num_batch} batches")


    def get_iterator(self):
        return iter(self.loader)


def resolve_data(config, paths):
    base = paths.datasets if config.data.root == "datasets" else paths.resolve(config.data.root)
    directory = base / config.data.directory
    version = Path(config.data.version)
    if version.is_absolute() or ".." in version.parts:
        raise ValueError("data.version must be relative")
    folder = directory / version
    info = json.loads((folder / "info.json").read_text())
    meta = json.loads((folder / "meta.json").read_text())
    shape = meta["data_shape"]
    stored = info["config"]
    is_od = config.model.id.startswith("od/")
    if len(shape) != (4 if is_od else 3):
        raise ValueError("Dataset layout does not match Flow/OD task")
    if is_od and shape[1] != shape[2]:
        raise ValueError("OD origin and destination axes must match")
    changes = {
        "data": {
            "node_num": shape[1],
            "seq_len": config.data.seq_len or stored["seq_length_x"],
            "horizon": config.data.horizon or stored["seq_length_y"],
            "input_dim": config.data.input_dim or (shape[1] if is_od else shape[-1]),
            "output_dim": config.data.output_dim or (shape[1] if is_od else shape[-1]),
        }
    }
    if is_od and (
        changes["data"]["input_dim"] != shape[1] or changes["data"]["output_dim"] != shape[1]
    ):
        raise ValueError("OD backbone dimensions must match node count")
    adjacency = base / config.data.adjacency
    if not adjacency.is_file():
        raise FileNotFoundError(adjacency)
    adj = np.load(adjacency, mmap_mode="r")
    if adj.shape != (shape[1], shape[1]):
        raise ValueError("Adjacency shape does not match nodes")
    return resolved_copy(config, changes), directory, adjacency, info


def _read(data_path, config):
    folder = Path(data_path) / config.data.version
    if (folder / "his.npy").exists():
        data = np.load(folder / "his.npy", mmap_mode="r")
    else:
        with np.load(folder / "his.npz") as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
    scaler = reconstruct_scaler(json.loads((folder / "meta.json").read_text()))
    return folder, data, scaler


def load_dataset(data_path, config, logger, drop=False):
    folder, data, scaler = _read(data_path, config)
    loaders = {}
    logger.info(
        f"Data shape: {data.shape}; shared float32 storage: {data.nbytes / 1024**2:.1f} MiB"
    )
    splits = ["train", "val", "test"]
    for split in splits:
        indices = np.load(folder / f"idx_{split}.npy")
        limit = getattr(config.runtime, f"max_{split}_samples", None)
        if limit is not None:
            indices = indices[:limit]
        ds = TimeSeriesDataset(data, indices, config.data.seq_len, config.data.horizon)
        loaders[f"{split}_loader"] = LoaderAdapter(
            ds,
            config.training.batch_size,
            shuffle=split == "train",
            drop_last=drop,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory,
            logger=logger,
            name=split,
        )
    return loaders, scaler


def load_dataset_series(data_path, config, logger, drop=False):
    """Full history plus exact forecast origins for causal statistical methods."""
    from data.temporal import training_end

    folder, data, scaler = _read(data_path, config)
    end = training_end(folder, config.data.horizon)
    idx = np.load(folder / "idx_test.npy")
    if config.runtime.max_train_samples is not None:
        raise ValueError("Statistical fitting uses a time prefix; max_train_samples is unsupported")
    if config.runtime.max_test_samples is not None:
        idx = idx[:config.runtime.max_test_samples]
    if not len(idx) or idx[0] < end - 1 or idx[-1] + config.data.horizon >= len(data):
        raise ValueError("Invalid statistical forecast origins")
    counts = scaler.inverse_transform(data, device="cpu")
    logger.info(f"Rolling statistical protocol: fit [0,{end}), {len(idx)} test origins")
    return {"series": counts, "origins": idx, "train_end": end}, scaler


def load_adj_from_numpy(path):
    return np.load(path)
