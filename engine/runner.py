"""Single-device experiment execution shared by the CLI and Python callers."""

from dataclasses import dataclass


from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
import json
import os
import random
import traceback
import numpy as np
import torch
import yaml
from config.loader import model_registry, resolved_copy
from config.paths import load_paths
from data.loader import load_dataset, resolve_data
from engine.logging import get_logger


@dataclass(frozen=True)
class RunResult:
    directory: Path
    checkpoint: Path | None
    metrics: dict
    skipped: bool = False


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def load_recipe(mid):
    entry = model_registry()[mid]
    return getattr(import_module(entry["module"]), entry.get("factory", "get_recipe"))()


def _optimizer(model, config):
    options = config.training.optimizer.to_dict()
    name = options.pop("name")
    if name == "none":
        return None
    if name not in ("Adam", "AdamW", "SGD"):
        raise ValueError(f"Unsupported optimizer: {name}")
    return getattr(torch.optim, name)(model.parameters(), **options)


def _scheduler(optimizer, config):
    options = config.training.scheduler.to_dict()
    name = options.pop("name")
    if name == "none" or optimizer is None:
        return None
    if name not in ("StepLR", "MultiStepLR", "CosineAnnealingLR"):
        raise ValueError(f"Unsupported scheduler: {name}")
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **options)


def build_engine(config, paths, directory, logger):
    from engine.trainer import BaseEngine, BaseEngine_OD
    from engine.calibration.quantile import CQR_Engine

    recipe = load_recipe(config.model.id)
    config, data_path, adj_path, info = resolve_data(config, paths)
    device = torch.device(config.runtime.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose runtime.device=cpu")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.set_device(device)
    config = resolved_copy(config, {"runtime": {"device": str(device)}})
    engine_cls = recipe.engine_cls or (BaseEngine_OD if recipe.od else BaseEngine)
    if config.calibration.mode != "no":
        if recipe.od:
            if not recipe.od_cqr:
                raise ValueError(f"{config.model.id} does not support OD conformal calibration")
            if config.runtime.mode == "train":
                raise ValueError(
                    "OD conformal calibration requires a trained checkpoint; use calibrate"
                )
            if config.calibration.method == "aci":
                from engine.calibration.od_aci import OD_ACI_Engine

                engine_cls = OD_ACI_Engine
            else:
                from engine.calibration.od_split import OD_CQR_Engine

                engine_cls = OD_CQR_Engine
        else:
            engine_cls = recipe.engine_quantile_cls or CQR_Engine
            config = resolved_copy(
                config,
                {
                    "data": {
                        "cqr_channels": config.data.output_dim,
                        "output_dim": config.data.output_dim * 3,
                    }
                },
            )
    loader_fn = recipe.load_data or load_dataset
    loaders, scaler = loader_fn(data_path, config, logger)
    context = (
        recipe.setup(config, str(data_path), str(adj_path), config.data.node_num, device, logger)
        if recipe.setup
        else {}
    )
    model = recipe.build_model(config, config.data.node_num, **(context or {})).to(device)
    if not recipe.od and config.calibration.mode != "no" and not model.cqr_compatible:
        raise ValueError(f"{config.model.id} cannot produce Flow quantiles")
    optimizer = _optimizer(model, config)
    kwargs = dict(
        device=device,
        model=model,
        dataloader=loaders,
        scaler=scaler,
        loss_fn=recipe.loss_fn,
        lrate=config.training.optimizer.lr,
        optimizer=optimizer,
        scheduler=_scheduler(optimizer, config),
        clip_grad_norm=config.training.clip_grad_norm,
        max_epochs=config.training.max_epochs,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        log_dir=str(directory),
        logger=logger,
        seed=config.training.seed,
        normalize=config.data.normalize,
        metric_list=recipe.metric_list or ["MAE", "MAPE", "MSE", "RMSE", "F1", "TZR"],
        init_weights=recipe.init_weights,
        config=config,
    )
    if recipe.engine_extras:
        kwargs.update(
            recipe.engine_extras(config) if callable(recipe.engine_extras) else recipe.engine_extras
        )
    engine = engine_cls(**kwargs)
    return config, engine, loaders, recipe, info


def _new_directory(paths, experiment):
    parent = paths.results / experiment
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for suffix in range(1000):
        directory = parent / f"{stamp}-{os.getpid()}-{suffix}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise RuntimeError("Cannot allocate unique run directory")


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def run_experiment(config, paths=None):
    paths = paths or load_paths()
    from engine.completion import find_completed

    completed = find_completed(config, paths)
    if completed is not None:
        return RunResult(*completed, skipped=True)
    paths.ensure_dataset_link()
    os.environ.setdefault("HF_HOME", str(paths.cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(paths.cache / "torch"))
    if config.runtime.threads:
        torch.set_num_threads(config.runtime.threads)
    set_seed(config.training.seed, config.runtime.deterministic)
    directory = _new_directory(paths, config.experiment.name)
    logger = get_logger(str(directory), f"popst.{directory.name}")
    _write_json(directory / "status.json", {"status": "running"})
    (directory / "config.requested.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False)
    )
    engine = None
    try:
        config, engine, loaders, recipe, info = build_engine(config, paths, directory, logger)
        (directory / "config.resolved.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False)
        )
        _write_json(
            directory / "data_manifest.json",
            {"dataset": config.data.to_dict(), "source_metadata": info},
        )
        _write_json(
            directory / "environment.json",
            {
                "python": os.sys.version,
                "device": config.runtime.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "paths": {key: str(value) for key, value in vars(paths).items()},
                "packages": {
                    n: metadata.version(n) for n in ["torch", "numpy", "pandas", "scipy", "PyYAML"]
                },
            },
        )
        if not config.runtime.quiet:
            logger.info(yaml.safe_dump(config.to_dict(), sort_keys=False))
        if config.runtime.mode == "train":
            if recipe.train_with_export:
                engine.train(config.output.predictions)
            else:
                engine.train()
        else:
            if config.runtime.checkpoint is None:
                raise ValueError("Evaluation/calibration requires runtime.checkpoint or --run")
            checkpoint = paths.resolve(config.runtime.checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if recipe.train_with_export:
                raise ValueError(
                    "Statistical models fit and evaluate with the train command; they do not load neural checkpoints"
                )
            engine.evaluate("test", str(checkpoint), config.output.predictions)
        metrics = getattr(engine.metric, "last_test", None)
        if config.runtime.mode == "train" and not config.training.evaluate_test:
            metrics = getattr(engine, "last_validation", None)
        if not metrics:
            raise RuntimeError("Run produced no final metrics")
        _write_json(directory / "metrics.json", metrics)
        if config.runtime.profile:
            from engine.profiling import profile_efficiency

            efficiency = profile_efficiency(engine, loaders, engine._device, logger)
            _write_json(directory / "efficiency.json", efficiency)
        checkpoint = directory / "best.pt"
        _write_json(
            directory / "status.json",
            {"status": "succeeded", "checkpoint": str(checkpoint) if checkpoint.exists() else None},
        )
        return RunResult(directory, checkpoint if checkpoint.exists() else None, metrics)
    except BaseException as exc:
        _write_json(
            directory / "status.json",
            {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
