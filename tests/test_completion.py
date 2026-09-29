"""Completed-run reuse must preserve experiment settings and reject partial results."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import yaml

from config.loader import load_config, resolved_copy
from config.paths import ROOT, load_paths
from engine.completion import find_completed


def save_completed(paths, config):
    directory = paths.results / "previous-name" / "run"
    directory.mkdir(parents=True)
    saved = config.to_dict()
    saved["runtime"].pop("skip_completed")  # Also recognize runs made before this feature.
    (directory / "config.requested.yaml").write_text(yaml.safe_dump(saved))
    (directory / "metrics.json").write_text(json.dumps({"MAE": 1.0}))
    (directory / "status.json").write_text(json.dumps({"status": "succeeded", "checkpoint": None}))
    return directory


@pytest.fixture
def completed(tmp_path):
    paths = replace(load_paths(), results=tmp_path / "results")
    config = load_config(model="flow/hl", dataset="dc_60min")
    return paths, config, save_completed(paths, config)


def test_single_run_reuses_before_building_engine(completed, monkeypatch):
    from engine.runner import run_experiment

    paths, config, directory = completed
    config = resolved_copy(config, {"experiment": {"name": "other"},
                                    "runtime": {"device": "cuda:2", "threads": 1}})

    def forbidden(*args, **kwargs):
        pytest.fail("Completed runs must not initialize data or launch training")

    monkeypatch.setattr("engine.runner.build_engine", forbidden)
    monkeypatch.setattr(type(paths), "ensure_dataset_link", forbidden)
    result = run_experiment(config, paths)
    assert result.skipped and result.directory == directory
    assert result.metrics == {"MAE": 1.0}


@pytest.mark.parametrize("changes", [
    {"training": {"batch_size": 64}},
    {"training": {"seed": 1}},
    {"data": {"version": "different"}},
    {"model": {"params": {"hidden_dim": 999}}},
    {"runtime": {"mode": "test"}},
    {"runtime": {"skip_completed": False}},
    {"runtime": {"profile": True}},
    {"output": {"predictions": True}},
])
def test_changed_settings_do_not_skip(completed, changes):
    paths, config, _ = completed
    assert find_completed(resolved_copy(config, changes), paths) is None


@pytest.mark.parametrize("file,contents", [
    ("status.json", '{"status": "failed"}'),
    ("status.json", '{"status": "running"}'),
    ("status.json", '{"status": "succeeded", "checkpoint": "/missing/best.pt"}'),
    ("status.json", "{"),
    ("metrics.json", "{}"),
    ("metrics.json", None),
    ("config.requested.yaml", None),
])
def test_incomplete_results_do_not_skip(completed, file, contents):
    paths, config, directory = completed
    if contents is None:
        (directory / file).unlink()
    else:
        (directory / file).write_text(contents)
    assert find_completed(config, paths) is None


def test_suite_skips_success_and_can_force_rerun(completed, tmp_path, monkeypatch):
    from engine import suite

    paths, config, directory = completed
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(config.to_dict()))
    suite_path = tmp_path / "suite.yaml"
    document = {"runs": [run_path.name], "devices": ["cpu"]}
    suite_path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(suite, "load_paths", lambda: paths)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", execute)
    assert suite.run_suite(suite_path) == 0
    assert not calls
    record = json.loads(next((paths.results / "suites").rglob("tasks.json")).read_text())[0]
    assert record["status"] == "skipped" and record["run"] == str(directory)
    document["skip_completed"] = False
    suite_path.write_text(yaml.safe_dump(document))
    assert suite.run_suite(suite_path) == 0
    assert len(calls) == 1 and "runtime.skip_completed=false" in calls[0]


def test_all_registered_models_and_runs_inherit_batch_size_128():
    from config.loader import model_registry

    for model in model_registry():
        dataset = "dc_od_60min" if model.startswith("od/") else "dc_60min"
        assert load_config(model=model, dataset=dataset).training.batch_size == 128
    for path in (ROOT / "config/runs").rglob("*.yaml"):
        assert load_config(path).training.batch_size == 128
