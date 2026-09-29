"""Model-specific construction and task behavior; no command-line parsing."""

from dataclasses import dataclass


from typing import Callable


@dataclass(frozen=True)
class ModelRecipe:
    build_model: Callable
    loss_fn: str = "MAE"
    engine_cls: type | None = None
    engine_quantile_cls: type | None = None
    init_weights: bool = False
    metric_list: list | None = None
    setup: Callable | None = None
    load_data: Callable | None = None
    engine_extras: dict | Callable | None = None
    train_with_export: bool = False
    od: bool = False
    od_cqr: bool = False
