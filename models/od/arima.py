import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor
from models.base import BaseModel
from engine.recipe import ModelRecipe
from engine.adapters.statistical import RollingStatisticalEngine
from data.loader import load_dataset_series
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


class ARIMA_(BaseModel):
    """ARIMA per OD cell and channel; fit once, then update with observed history."""

    def __init__(self, order=(6, 0, 0), n_threads=16, **args):
        super().__init__(**args)
        self.order = order
        self.n_threads = n_threads

    def forecast_origins(self, series, train_end, origins, horizon):
        return arima_rolling(series, train_end, origins, horizon, self.order, workers=self.n_threads)


def build_model(config, node_num, **ctx):
    return ARIMA_(
        order=config.model.params.order,
        n_threads=config.model.params.n_threads,
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


def arima_rolling(series, train_end, origins, horizon, order, workers=1):
    shape = series.shape[1:]
    flat = np.asarray(series, dtype=np.float64).reshape(len(series), -1)

    def cell(k):
        values = flat[:, k]
        train = values[:train_end]
        if np.ptp(train) == 0:
            return np.full((len(origins), horizon), train[-1], dtype=np.float32)
        model = ARIMA(train, order=order)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                fitted = model.fit()
                rows, consumed = [], train_end
                for origin in origins:
                    if origin + 1 > consumed:
                        fitted = fitted.extend(values[consumed:origin + 1])
                        consumed = origin + 1
                    rows.append(np.asarray(fitted.forecast(horizon)))
            except Exception as exc:
                raise RuntimeError(f"Statistical fit/forecast failed for OD cell {k}: {exc}") from exc
        result = np.asarray(rows, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError(f"Non-finite statistical forecast for OD cell {k}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        forecasts = list(pool.map(cell, range(flat.shape[1])))
    return np.stack(forecasts, axis=-1).reshape(len(origins), horizon, *shape)
