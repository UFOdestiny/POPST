"""Seasonal naive: repeat the most recent observed daily or weekly cycle."""

import numpy as np
from models.base import BaseModel
from engine.recipe import ModelRecipe
from engine.adapters.statistical import RollingStatisticalEngine
from data.loader import load_dataset_series
from data.temporal import steps_per_day


class SeasonalNaive(BaseModel):
    def __init__(self, period, **kwargs):
        super().__init__(**kwargs)
        self.period = period

    def forecast_origins(self, series, train_end, origins, horizon):
        # h > period repeats the last available season, never future labels.
        h = np.arange(1, horizon + 1)
        offsets = h - ((h + self.period - 1) // self.period) * self.period
        sources = origins[:, None] + offsets
        if sources.min() < 0:
            raise ValueError("Seasonal naive requires one full observed season")
        return np.asarray(series[sources], dtype=np.float32)


def build_model(config, node_num, **ctx):
    return SeasonalNaive(
        period=steps_per_day(config.data.frequency) * config.model.params.days,
        node_num=node_num, input_dim=node_num, output_dim=node_num,
        seq_len=config.data.seq_len, horizon=config.data.horizon,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, od=True, train_with_export=True,
                       engine_cls=RollingStatisticalEngine, load_data=load_dataset_series)
