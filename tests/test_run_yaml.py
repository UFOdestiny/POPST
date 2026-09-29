"""Run YAML dispatch and suite orchestration without scheduling real jobs."""

from types import SimpleNamespace

import pytest
import yaml

from config.loader import load_config
from config.paths import ROOT
from run import main


def write_run(tmp_path, name, **sections):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "flow/hl"},
                "data": {"id": "dc_60min"},
                **sections,
            }
        )
    )
    return path


@pytest.mark.parametrize("mode", ["train", "test", "calibrate"])
def test_yaml_dispatch_preserves_mode(tmp_path, monkeypatch, mode):
    path = write_run(
        tmp_path, mode, runtime={"mode": mode, "device": "cpu"}, calibration={"mode": "horizon"}
    )
    received = []

    def execute(config):
        received.append(config)
        return SimpleNamespace(directory=tmp_path, checkpoint=None, metrics={})

    monkeypatch.setattr("engine.runner.run_experiment", execute)
    assert main([str(path)]) == 0
    assert received[0].runtime.mode == mode
    assert received[0].runtime.device == "cpu"
    assert main(["--config", str(path), "--dry-run"]) == 0
    assert len(received) == 1


@pytest.mark.parametrize("prefix", [[], ["--config"]])
@pytest.mark.parametrize("preview", [False, True])
def test_suite_yaml_dispatch(tmp_path, monkeypatch, prefix, preview):
    # Use a path outside config/suites to ensure dispatch depends on content.
    path = tmp_path / "batch.yml"
    path.write_text(yaml.safe_dump({"runs": ["run.yaml"], "devices": ["cuda:0"]}))
    received = []

    def execute(config, dry_run=False):
        received.append((config, dry_run))
        return 0 if dry_run else 7

    monkeypatch.setattr("engine.suite.run_suite", execute)
    result = main([*prefix, str(path), *(["--dry-run"] if preview else [])])
    assert received == [(path, preview)]
    assert result == (0 if preview else 7)


def test_direct_suite_preview_does_not_execute(tmp_path, monkeypatch, capsys):
    import json
    run = write_run(tmp_path, "run")
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump({
        "runs": [run.name], "matrix": {"training.seed": [2026]},
        "devices": ["cuda:0", "cuda:1", "cuda:2"],
    }))

    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not execute a workload")

    monkeypatch.setattr("engine.suite.subprocess.run", forbidden)
    monkeypatch.setattr("engine.suite.load_paths", forbidden)
    assert main([str(path), "--dry-run"]) == 0
    task = json.loads(capsys.readouterr().out)
    assert "training.seed=2026" in task["command"]
    assert task["devices"] == ["cuda:0", "cuda:1", "cuda:2"]


def test_checkpoint_selection_and_overrides(tmp_path, monkeypatch):
    import shutil
    path = write_run(tmp_path, "run")
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(path, source / "config.requested.yaml")
    received = []

    def execute(config):
        received.append(config)
        return SimpleNamespace(directory=tmp_path, checkpoint=None, metrics={})

    monkeypatch.setattr("engine.runner.run_experiment", execute)
    assert main(["--run", str(source), "--set", "runtime.mode=test", "--device", "cpu"]) == 0
    cfg = received[0]
    assert cfg.runtime.mode == "test" and cfg.runtime.device == "cpu"
    assert cfg.runtime.checkpoint == str(source / "best.pt")
    calibrated = write_run(tmp_path, "cal", runtime={"mode": "calibrate"}, calibration={"mode": "horizon"})
    assert main([str(calibrated), "--run", str(source)]) == 0
    assert received[1].runtime.mode == "calibrate"


def test_conflicting_or_ignored_arguments_are_rejected(tmp_path, capsys):
    path = write_run(tmp_path, "run")
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--config", str(path)])
    assert exc.value.code == 2
    assert "Specify the YAML path once" in capsys.readouterr().err
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"runs": [path.name]}))
    with pytest.raises(SystemExit) as exc:
        main([str(suite), "--device", "cuda:2", "--dry-run"])
    assert exc.value.code == 2
    assert "apply to single runs" in capsys.readouterr().err



def test_all_run_and_suite_paths():
    from engine.suite import expand_suite

    paths = list((ROOT / "config/runs").rglob("*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path)
        assert path.parent.name == cfg.data.id
    for path in (ROOT / "config/suites").glob("*.yaml"):
        assert list(expand_suite(path))


def test_suite_concurrent_devices_and_failure(tmp_path, monkeypatch):
    import json
    from threading import Barrier
    from engine import suite

    run = write_run(tmp_path, "run")
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({
        "runs": [run.name], "matrix": {"training.seed": [1, 2, 3]},
        "devices": "all",
    }))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)
    barrier = Barrier(3)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        barrier.wait(timeout=10)  # All three must be started before any finishes.
        return SimpleNamespace(returncode=7 if command[-1] == "cuda:1" else 0)

    monkeypatch.setattr(suite, "load_paths", lambda: SimpleNamespace(results=tmp_path / "results"))
    monkeypatch.setattr(suite.subprocess, "run", execute)
    assert suite.run_suite(path) == 1
    assert {command[-1] for command in calls} == {"cuda:0", "cuda:1", "cuda:2"}
    records = json.loads(next((tmp_path / "results").rglob("tasks.json")).read_text())
    assert sorted(row["status"] for row in records) == ["failed", "succeeded", "succeeded"]


def test_suite_preserves_calibration_mode_and_rejects_missing_checkpoint(tmp_path):
    from engine.suite import expand_suite, run_suite
    run = write_run(tmp_path, "cal", runtime={"mode": "calibrate"})
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"runs": [run.name], "devices": ["cpu"]}))
    command, cfg = next(expand_suite(path))
    assert cfg.runtime.mode == "calibrate"
    assert "train" not in command
    with pytest.raises(ValueError, match="exact runtime.checkpoint"):
        run_suite(path)


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("value", ["all", ["all"]])
def test_suite_discovers_visible_devices(monkeypatch, count, value):
    from engine.suite import resolve_devices
    monkeypatch.setattr("torch.cuda.device_count", lambda: count)
    assert resolve_devices(value) == [f"cuda:{i}" for i in range(count)]


def test_suite_device_selection_and_no_gpu(monkeypatch):
    from engine.suite import resolve_devices
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="visible CUDA GPU"):
        resolve_devices("all")
    assert resolve_devices(["cuda:2", "cuda:0"]) == ["cuda:2", "cuda:0"]
    assert resolve_devices(["cuda"]) == ["cuda:0"]
    assert resolve_devices(["cpu"]) == ["cpu"]
    for invalid in ([], "cuda:0", ["all", "cuda:0"], ["cuda", "cuda:0"], [3]):
        with pytest.raises(ValueError):
            resolve_devices(invalid)


def test_provided_suite_contracts():
    from engine.suite import expand_suite
    suites = ROOT / "config/suites"
    for path in suites.glob("*.yaml"):
        doc = yaml.safe_load(path.read_text())
        assert doc["devices"] == "all"
        for _, cfg in expand_suite(path):
            assert cfg.runtime.device != "cpu"
            assert cfg.training.seed == 2026
            assert cfg.training.batch_size == 128
        if path.stem.endswith(("_flow", "_od")):
            configs = [cfg for _, cfg in expand_suite(path)]
            assert len(configs) == 19
            assert all(cfg.model.id != "flow/transformer" for cfg in configs)
    baseline = [cfg for _, cfg in expand_suite(suites / "od_baselines.yaml")]
    own = [cfg for _, cfg in expand_suite(suites / "od_zeropdr.yaml")]
    assert len(baseline) == 60 and len(own) == 54
    assert {cfg.data.id for cfg in baseline} == {cfg.data.id for cfg in own}
    assert {cfg.model.id for cfg in own} == {
        f"od/{name}" for name in ("pdr", "pdr_reg", "pdr_no_context", "pdr_no_zone_embed",
        "pdr_no_spatial", "pdr_no_moe", "pdr_reg_gau", "pdr_reg_lap", "pdr_reg_t")
    }
    assert all(cfg.runtime.mode == "train" for cfg in own)


def test_legacy_scheduler_metadata_does_not_break_checkpoint_loading(tmp_path):
    path = write_run(tmp_path, "old", slurm={"gpus": 3, "mem": "64G"})
    cfg = load_config(path)
    assert "slurm" not in cfg
    with pytest.raises(ValueError, match="Unknown configuration field: slurm"):
        load_config(path, overrides=["slurm.gpus=1"])
