import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from base.model import BaseModel


class ARIMA_(BaseModel):
    """ARIMA per OD-cell-and-channel forecasting (multi-threaded).

    Fits one ARIMA(p,d,q) to each ``(origin, destination, mobility-channel)``
    time series of the training split and forecasts the test horizon.  Channel
    is just a third task index, so all mobility channels are handled uniformly.
    """

    def __init__(self, order=(6, 0, 0), n_threads=16, **args):
        super().__init__(**args)
        self.order = order
        self.n_threads = n_threads

    def _fit_forecast(self, ts, steps, idx):
        try:
            model = ARIMA(ts, order=self.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return (idx, forecast)
        except Exception as e:
            print(f"ARIMA failed at {idx}: {e}")
            return (idx, np.full(steps, np.nan))

    def forward(self, X_train, Y_test_len):
        """X_train ``(T_train, N, N, D)`` -> Y_pred ``(Y_test_len, N, N, D)``.

        Also accepts a 3-D ``(T_train, N, N)`` array (single channel)."""
        X_train = np.asarray(X_train)
        single = X_train.ndim == 3
        if single:
            X_train = X_train[..., None]

        T_train, N, _, D = X_train.shape
        Y_pred = np.zeros((Y_test_len, N, N, D), dtype=np.float32)

        tasks = [
            ((i, j, c), X_train[:, i, j, c])
            for i in range(N) for j in range(N) for c in range(D)
        ]

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(self._fit_forecast, ts, Y_test_len, idx): idx
                for idx, ts in tasks
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc="ARIMA"):
                (i, j, c), forecast = f.result()
                Y_pred[:, i, j, c] = forecast

        return Y_pred[..., 0] if single else Y_pred
