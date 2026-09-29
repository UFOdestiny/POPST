import numpy as np
from sklearn.decomposition import TruncatedSVD
from models.base import BaseModel
from engine.recipe import ModelRecipe
from engine.adapters.statistical import RollingStatisticalEngine
from data.loader import load_dataset_series


class VAR(BaseModel):
    """Training-only low-rank VAR, fitted independently for each mobility channel."""

    def __init__(self, node_num, input_dim, output_dim, k=6, lags=6, seed=2025, **args):
        super().__init__(node_num, input_dim, output_dim, **args)
        self.k = k
        self.lags = lags
        self.seed = seed

    def forecast_origins(self, series, train_end, origins, horizon):
        return var_rolling(series, train_end, origins, horizon, self.k, self.lags, self.seed)


def build_model(config, node_num, **ctx):
    return VAR(
        k=config.model.params.k,
        lags=config.model.params.lags,
        seed=config.training.seed,
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


def var_rolling(series, train_end, origins, horizon, rank, lags, seed):
    from statsmodels.tsa.api import VAR

    n, c = series.shape[1], series.shape[-1]
    result = np.empty((len(origins), horizon, n, n, c), dtype=np.float32)
    for channel in range(c):
        flat = series[..., channel].reshape(len(series), -1)
        train = flat[:train_end]
        if np.ptp(train, axis=0).max() == 0:
            result[..., channel] = train[-1].reshape(1, 1, n, n)
            continue
        k = min(rank, flat.shape[1], train_end - 1)
        if k < 2 or train_end <= (k + 1) * lags + 1:
            raise ValueError("Not enough training observations for requested low-rank VAR")
        svd = TruncatedSVD(n_components=k, random_state=seed)
        factors = svd.fit_transform(train)
        fitted = VAR(factors).fit(maxlags=lags, trend="n")
        if fitted.k_ar < 1:
            raise RuntimeError("VAR selected no autoregressive lag")
        for row, origin in enumerate(origins):
            history = svd.transform(flat[origin - fitted.k_ar + 1:origin + 1])
            forecast = fitted.forecast(history, steps=horizon)
            result[row, ..., channel] = svd.inverse_transform(forecast).reshape(horizon, n, n)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite low-rank VAR forecast")
    return result
