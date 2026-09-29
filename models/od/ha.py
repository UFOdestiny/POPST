from models.base import BaseModel
import numpy as np
from engine.recipe import ModelRecipe
from engine.adapters.statistical import RollingStatisticalEngine
from data.loader import load_dataset_series


class HA(BaseModel):
    """Historical Average: predict each test step as the mean of the preceding
    ``step`` observations.  Shape-agnostic — operates on whole ``(T, N, N, D)``
    arrays, so all mobility channels are handled at once."""

    def __init__(self, step=6, **args):
        super(HA, self).__init__(**args)
        self.step = step

    def forecast_origins(self, series, train_end, origins, horizon):
        if origins.min() + 1 < self.step:
            raise ValueError("HA requires a complete observed history window")
        points = np.stack([series[t - self.step + 1:t + 1].mean(axis=0) for t in origins])
        return np.repeat(points[:, None], horizon, axis=1).astype(np.float32)


def build_model(config, node_num, **ctx):
    return HA(
        step=config.model.params.step,
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
    )


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        od=True,
        engine_cls=RollingStatisticalEngine,
        load_data=load_dataset_series,
        train_with_export=True,
    )
