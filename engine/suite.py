"""Batch lists of run files; one independent process per configured device."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
from queue import Queue
import re
from threading import Lock
import subprocess
import sys

from config.loader import read_yaml, load_config
from config.paths import ROOT, load_paths
from engine.completion import find_completed


def resolve_devices(value):
    """Expand all visible CUDA devices without changing CUDA_VISIBLE_DEVICES."""
    if value == "all" or value == ["all"]:
        import torch

        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("devices: all requires at least one visible CUDA GPU")
        return [f"cuda:{index}" for index in range(count)]
    if not isinstance(value, list) or not value:
        raise ValueError("devices must be 'all' or a nonempty device list")
    if any(not isinstance(d, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", d) for d in value):
        raise ValueError("Use devices: all or a list of cpu/cuda:N devices")
    devices = ["cuda:0" if d == "cuda" else d for d in value]
    if len(set(devices)) != len(devices):
        raise ValueError("devices must not contain duplicate devices")
    return devices


def expand_suite(path):
    path = Path(path).resolve()
    doc = read_yaml(path)
    unknown = set(doc) - {"runs", "matrix", "overrides", "devices", "skip_completed"}
    if unknown:
        raise ValueError(f"Unknown suite fields: {sorted(unknown)}")
    if "skip_completed" in doc and not isinstance(doc["skip_completed"], bool):
        raise TypeError("suite.skip_completed must be a boolean")
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("suite.runs must be a nonempty list of run YAML paths")
    matrix = doc.get("matrix", {})
    if any(not isinstance(v, list) or not v for v in matrix.values()):
        raise ValueError("Suite dimensions must be nonempty lists")
    keys = list(matrix)
    for run in runs:
        base = (path.parent / run).resolve()
        for values in itertools.product(*(matrix[k] for k in keys)):
            settings = {**doc.get("overrides", {}), **dict(zip(keys, values))}
            if "skip_completed" in doc:
                settings.setdefault("runtime.skip_completed", doc["skip_completed"])
            overrides = [f"{k}={json.dumps(v)}" for k, v in settings.items()]
            cfg = load_config(base, overrides)
            if "experiment.name" not in settings:
                overrides.append(f"experiment.name={path.stem}_{base.stem}_{cfg.data.id}_s{cfg.training.seed}")
                cfg = load_config(base, overrides)
            yield ([sys.executable, str(ROOT / "run.py"), str(base),
                    *[part for override in overrides for part in ["--set", override]]], cfg)


def run_suite(path, dry_run=False):
    devices = resolve_devices(read_yaml(path).get("devices", ["cuda:0"]))
    jobs = list(expand_suite(path))
    if dry_run:
        for command, cfg in jobs:
            print(json.dumps({"command": command, "devices": ["cpu"] if cfg.runtime.device == "cpu" else devices}))
        return 0
    # Calibration never guesses a checkpoint, and is rejected before other jobs start.
    for _, cfg in jobs:
        if cfg.runtime.mode != "train" and cfg.runtime.checkpoint is None:
            raise ValueError("Every evaluation/calibration run needs an exact runtime.checkpoint")
    from engine.runner import _new_directory, _write_json

    paths = load_paths()
    output = _new_directory(paths, "suites")
    slots = Queue()
    for device in devices:
        slots.put(device)
    records = [{"index": i, "model": cfg.model.id, "dataset": cfg.data.id,
                "seed": cfg.training.seed,
                "status": "pending"} for i, (_, cfg) in enumerate(jobs)]
    _write_json(output / "tasks.json", records)
    record_lock = Lock()
    print(f"Suite output: {output}", flush=True)

    def execute(index, command, cfg):
        completed = find_completed(cfg, paths)
        if completed is not None:
            return {**records[index], "status": "skipped", "run": str(completed[0]),
                    "returncode": 0}
        slot = slots.get()
        try:
            device = "cpu" if cfg.runtime.device == "cpu" else slot
            command = [*command, "--device", device]
            log_path = output / f"{index:04d}.log"
            with record_lock:
                records[index].update(status="running", device=device, command=command, log=str(log_path))
                _write_json(output / "tasks.json", records)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            return {**records[index], "device": device, "command": command,
                    "status": "succeeded" if result.returncode == 0 else "failed",
                    "returncode": result.returncode, "log": str(log_path)}
        finally:
            slots.put(slot)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {pool.submit(execute, i, *job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            i = futures[future]
            with record_lock:
                try:
                    records[i] = future.result()
                except Exception as exc:
                    records[i].update(status="failed", returncode=1, error=str(exc))
                _write_json(output / "tasks.json", records)
            print(f"[{i + 1}/{len(jobs)}] {records[i]['model']}: {records[i]['status']}", flush=True)
    return int(any(r["status"] not in ("succeeded", "skipped") for r in records))
