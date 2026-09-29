import json
import numpy as np
import pandas as pd
import pytest
import torch
from config.loader import load_config, model_registry, resolved_copy
from config.paths import load_paths, ROOT
from data.loader import TimeSeriesDataset
from data.preprocessing import MinMaxScaler, _reorder_to_time_first
from data.spatial import make_step_idx_fn, accumulate_od


def test_all_model_and_data_descriptors():
    assert {"od/zero", "od/persistence", "od/seasonal", "od/odmixer"} <= set(model_registry())
    assert all("stllm" not in name for name in model_registry())
    for name in model_registry():
        cfg = load_config(
            model=name, dataset="dc_od_60min" if name.startswith("od/") else "dc_60min"
        )
        assert cfg.model.id == name
    for p in (ROOT / "config/datasets").glob("*.yaml"):
        cfg = load_config(model="od/hl" if "_od_" in p.stem else "flow/hl", dataset=p.stem)
        assert (load_paths().datasets / cfg.data.directory).is_dir()


def test_config_isolated_and_strict(tmp_path):
    a = load_config(model="flow/hl", dataset="dc_60min")
    with pytest.raises(TypeError):
        a.runtime.device = "cpu"
    with pytest.raises(ValueError, match="Unknown"):
        load_config(model="flow/hl", dataset="dc_60min", overrides=["training.batch_szie=1"])
    with pytest.raises(TypeError):
        load_config(model="flow/hl", dataset="dc_60min", overrides=["training.batch_size=true"])
    b = resolved_copy(a, {"runtime": {"device": "cpu"}})
    assert a.runtime.device == "cuda" and b.runtime.device == "cpu"
    c = tmp_path / "cycle.yaml"
    c.write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(c)


def test_paths_independent_of_cwd(tmp_path, monkeypatch):
    before = load_paths()
    monkeypatch.chdir(tmp_path)
    after = load_paths()
    assert before == after
    assert before.datasets.is_symlink()
    assert before.results.parent == ROOT


def test_windows_and_shared_storage():
    data = np.arange(100 * 3 * 2, dtype=np.float32).reshape(100, 3, 2)
    train = TimeSeriesDataset(data, np.array([11, 12]), 12, 3)
    val = TimeSeriesDataset(data, np.array([70, 71]), 12, 3)
    assert train.data is val.data
    x, y = train[0]
    np.testing.assert_array_equal(x.numpy(), data[:12])
    np.testing.assert_array_equal(y.numpy(), data[12:15])
    with pytest.raises(ValueError):
        TimeSeriesDataset(data, [98], 12, 3)


@pytest.mark.parametrize("log1p", [False, True])
@pytest.mark.parametrize("tensor_fit", [False, True])
def test_scaler_roundtrip_and_zero_channel(log1p, tensor_fit):
    raw = np.array([[0, 0], [1, 0], [8, 0]], dtype=np.float32)
    scaler = MinMaxScaler(use_log1p=log1p).fit(
        torch.from_numpy(raw) if tensor_fit else raw, per_channel=True
    )
    z = scaler.transform(raw)
    np.testing.assert_allclose(scaler.inverse_transform(z), raw, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        scaler.inverse_transform(torch.from_numpy(z), device=torch.device("cpu")),
        torch.from_numpy(raw),
    )
    assert np.all(z[:, 1] == 0)


def test_od_axis_order_and_count_binning():
    raw = np.arange(2 * 2 * 3 * 7).reshape(2, 2, 3, 7)
    actual = _reorder_to_time_first(raw, "NNDT")
    np.testing.assert_array_equal(actual, raw.transpose(3, 0, 1, 2))
    origin = np.array([0, 0, 1, -1])
    dest = np.array([1, 1, 0, 0])
    steps = np.array([0, 0, 1, 0])
    counts = np.zeros((2, 2, 2), dtype=np.int64)
    kept = accumulate_od(origin, dest, steps, np.ones(4, dtype=bool), counts, 2)
    assert kept == 3 and counts.sum() == 3 and counts[0, 1, 0] == 2


def test_time_grid_half_open_boundaries():
    fn = make_step_idx_fn("60min", 2, pd.Timestamp("2025-01-01"))
    index, valid = fn(
        pd.Series(["2025-01-01 00:59", "2025-01-01 01:00", "2025-01-01 02:00", "bad"])
    )
    assert valid.tolist() == [True, True, False, False]
    assert index[:2].tolist() == [0, 1]


def test_dataset_index_bounds_all_versions():
    from config.loader import read_yaml

    for p in (ROOT / "config/datasets").glob("*.yaml"):
        spec = read_yaml(p)["data"]
        for folder in (load_paths().datasets / spec["directory"]).iterdir():
            if not (folder / "info.json").is_file():
                continue
            info = json.loads((folder / "info.json").read_text())
            t = info["raw_data"]["shape"][0]
            for split in ["train", "val", "test"]:
                idx = np.load(folder / f"idx_{split}.npy")
                assert len(idx) > 0 and idx.min() >= info["config"]["seq_length_x"] - 1
                assert idx.max() + info["config"]["seq_length_y"] < t


def test_optimizer_scheduler_selection():
    cfg = load_config(
        model="flow/hl",
        dataset="dc_60min",
        overrides=[
            "training.optimizer.name=SGD",
            "training.optimizer.momentum=0.9",
            "training.scheduler.name=CosineAnnealingLR",
            "training.scheduler.T_max=10",
        ],
    )
    assert cfg.training.optimizer.momentum == 0.9 and cfg.training.scheduler.T_max == 10
    with pytest.raises(ValueError):
        load_config(model="flow/hl", dataset="dc_60min", overrides=["experiment.name=.."])
    with pytest.raises(ValueError):
        load_config(
            model="flow/hl", dataset="dc_60min", overrides=["data.directory=../elsewhere"]
        )
