"""Exercise the public execution contract, including exact checkpoint replay."""

from dataclasses import replace
import json
import numpy as np
import pytest
import requests
from config.loader import load_config, resolved_copy
from config.paths import load_paths
from data.prepare import prepare_array
from engine.runner import run_experiment


def test_train_reload_and_failure_status(tmp_path):
    raw = tmp_path / "raw.npy"
    np.save(raw, np.random.default_rng(2026).poisson(3, size=(3, 2, 96)).astype(np.float32))
    fixture = tmp_path / "fixture"
    prepare_array(raw, "dc_60min", "2025_12to1", fixture, per_channel=True, log1p=True)
    np.save(fixture / "dc_60min/dc.npy", np.eye(3, dtype=np.float32))
    paths = replace(load_paths(), datasets=fixture, source=fixture, results=tmp_path / "training")
    cfg = load_config(model="flow/hl", dataset="dc_60min", overrides=[
        "runtime.device=cpu", "runtime.threads=2", "runtime.quiet=true",
        "training.max_epochs=1", "training.batch_size=8", "output.predictions=true",
        "runtime.max_train_samples=16", "runtime.max_val_samples=8", "runtime.max_test_samples=8",
    ])
    result = run_experiment(cfg, paths)
    assert result.checkpoint.is_file()
    paths = replace(
        load_paths(),
        datasets=tmp_path / "fixture",
        source=tmp_path / "fixture",
        results=tmp_path / "replay",
    )
    cfg = load_config(
        result.directory / "config.requested.yaml", overrides=["runtime.profile=true"]
    )
    cfg = resolved_copy(cfg, {"runtime": {"mode": "test", "checkpoint": str(result.checkpoint)}})
    replay = run_experiment(cfg, paths)
    assert replay.metrics == result.metrics
    assert json.loads((replay.directory / "efficiency.json").read_text())["gpu_memory"] is None
    np.testing.assert_array_equal(
        np.load(next(result.directory.glob("*res.npy"))),
        np.load(next(replay.directory.glob("*res.npy"))),
    )
    with pytest.raises(FileNotFoundError):
        run_experiment(
            resolved_copy(cfg, {"runtime": {"checkpoint": str(tmp_path / "missing.pt")}}), paths
        )
    statuses = [json.loads(p.read_text())["status"] for p in paths.results.rglob("status.json")]
    assert sorted(statuses) == ["failed", "succeeded"]


def test_paged_download_resume_and_http_error(tmp_path, monkeypatch):
    from data import downloads

    calls = []

    class Response:
        text = "time,id\n2025-01-01,1\n2025-01-02,2\n"

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, params, timeout):
            calls.append(params)
            response = Response()
            if len(calls) > 1:
                response.text = "time,id\n"
            return response

    monkeypatch.setattr(downloads, "_make_session", Session)
    path = tmp_path / "rows.csv"
    downloads.download_from_api("https://fixture.invalid", "1=1", path, "time", "id", page_size=2)
    assert len(path.read_text().splitlines()) == 3
    assert "time > '2025-01-02'" in calls[1]["$where"]
    downloads.download_from_api("https://fixture.invalid", "1=1", path, "time", "id", page_size=2)
    assert len(path.read_text().splitlines()) == 3
    assert "time > '2025-01-02'" in calls[2]["$where"]

    def fail(self):
        raise requests.HTTPError("503")

    monkeypatch.setattr(Response, "raise_for_status", fail)
    with pytest.raises(requests.HTTPError):
        downloads.download_from_api("https://fixture.invalid", "1=1", path, "time", "id")


@pytest.mark.parametrize("model", ["zero", "persistence", "seasonal", "ha", "arima", "sarima", "var"])
def test_statistical_training_and_export(tmp_path, model):
    raw = tmp_path / "raw.npy"
    np.save(raw, np.full((2, 2, 1, 120), 3, dtype=np.float32))
    fixture = tmp_path / "fixture"
    prepare_array(raw, "dc_od_60min", "2025_12to1", fixture, fmt="NNDT", per_channel=True, log1p=True)
    np.save(fixture / "dc_od_60min/dc.npy", np.eye(2, dtype=np.float32))
    paths = replace(load_paths(), datasets=fixture, source=fixture, results=tmp_path / "results")
    cfg = load_config(model=f"od/{model}", dataset="dc_od_60min", overrides=[
        "runtime.device=cpu", "runtime.threads=2", "runtime.quiet=true",
        "runtime.max_test_samples=4", "output.predictions=true",
    ])
    result = run_experiment(cfg, paths)
    assert result.checkpoint is None
    assert result.metrics and next(result.directory.glob("*res.npy")).is_file()


@pytest.mark.parametrize("model", ["odmixer", "stpro", "od_ced"])
def test_author_od_adapters_train_and_replay(tmp_path, model):
    raw = tmp_path / "raw.npy"
    np.save(raw, np.random.default_rng(2026).poisson(3, size=(4, 4, 2, 160)).astype(np.float32))
    fixture = tmp_path / "fixture"
    prepare_array(raw, "dc_od_60min", "2025_12to1", fixture, fmt="NNDT",
                  seq_length_y=3, per_channel=True, log1p=True)
    np.save(fixture / "dc_od_60min/dc.npy", np.eye(4, dtype=np.float32))
    paths = replace(load_paths(), datasets=fixture, source=fixture, results=tmp_path / "training")
    cfg = load_config(model=f"od/{model}", dataset="dc_od_60min", overrides=[
        "runtime.device=cpu", "runtime.threads=2", "runtime.quiet=true",
        "training.max_epochs=1", "training.batch_size=8", "output.predictions=true",
        "output.test_inputs=true", "runtime.max_train_samples=16",
        "runtime.max_val_samples=8", "runtime.max_test_samples=8",
    ])
    result = run_experiment(cfg, paths)
    assert result.checkpoint.is_file() and result.metrics
    replay_cfg = resolved_copy(load_config(result.directory / "config.requested.yaml"), {
        "runtime": {"mode": "test", "checkpoint": str(result.checkpoint), "profile": True}})
    replay = run_experiment(replay_cfg, replace(paths, results=tmp_path / "replay"))
    assert replay.metrics == result.metrics
    prediction = np.load(next(result.directory.glob("*res.npy")))
    np.testing.assert_array_equal(prediction, np.load(next(replay.directory.glob("*res.npy"))))
    assert prediction.shape[-4:] == (3, 4, 4, 2)
