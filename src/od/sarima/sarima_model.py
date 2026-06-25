import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from base.model import BaseModel
import warnings

warnings.filterwarnings("ignore")  # suppress SARIMAX FutureWarnings


class SARIMA_(BaseModel):
    """SARIMA per OD-cell-and-channel forecasting (multi-threaded).

    Fits one SARIMAX(order[, seasonal_order]) per
    ``(origin, destination, mobility-channel)`` training series and forecasts
    the test horizon.  Channel is a third task index.
    """

    def __init__(self, order=(6, 0, 0), seasonal_order=(0, 0, 0, 0), n_threads=8, **args):
        super().__init__(**args)
        self.order = order
        self.seasonal_order = seasonal_order
        self.n_threads = n_threads

    def _fit_forecast(self, ts, steps, idx):
        try:
            model = SARIMA(ts, order=self.order, seasonal_order=self.seasonal_order)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=steps)
            return (idx, forecast)
        except Exception as e:
            print(f"SARIMA failed at {idx}: {e}")
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
            for f in tqdm(as_completed(futures), total=len(futures), desc="SARIMA"):
                (i, j, c), forecast = f.result()
                Y_pred[:, i, j, c] = forecast

        return Y_pred[..., 0] if single else Y_pred
