"""Last-observation forecast at each origin, held over the whole horizon."""

import numpy as np
from models.base import BaseModel
from engine.recipe import ModelRecipe
from engine.adapters.statistical import RollingStatisticalEngine
from data.loader import load_dataset_series


class Persistence(BaseModel):
    def forecast_origins(self, series, train_end, origins, horizon):
        return np.repeat(series[origins, None], horizon, axis=1).astype(np.float32)


def build_model(config, node_num, **ctx):
    return Persistence(node_num, node_num, node_num, config.data.seq_len, config.data.horizon)


def get_recipe():
    return ModelRecipe(build_model=build_model, od=True, train_with_export=True,
                       engine_cls=RollingStatisticalEngine, load_data=load_dataset_series)
