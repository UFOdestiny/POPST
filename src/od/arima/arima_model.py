import torch
import torch.nn as nn
from base.model import BaseModel
from statsmodels.tsa.arima.model import ARIMA as A
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")  # ARIMA 的未来警告略过

class ARIMA(BaseModel):
    def __init__(self, node_num, input_dim, output_dim, arima_order, **args):
        super(ARIMA, self).__init__(node_num, input_dim, output_dim)
        self.arima_order = arima_order
        self.max_workers = 8
    
    def _fit_single_series(self, series_info):
        """
        Fit ARIMA to a single time series and forecast.
        series_info: (b, i, j, series)
        """
        b, i, j, series = series_info
        print(b, i, j)
        try:
            model = ARIMA(series, order=self.order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            return (b, i, j, forecast[0])
        except Exception:
            return (b, i, j, series[-1])  # fallback to last value


    def forward(self, input, label=None):  # (b, t, n, f)
        # print(input.shape)
        B, T, N, F = input.shape
        output = torch.zeros(B, self.horizon, N, F)

        # Prepare series list
        series_list = [
            (b, i, j, input[b, :, i, j])
            for b in range(B)
            for i in range(N)
            for j in range(N)
        ]

        # Run multithreaded ARIMA fitting
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._fit_single_series, series_info) for series_info in series_list]
            for future in as_completed(futures):
                b, i, j, val = future.result()
                output[b, 0, i, j] = val
        # print("output~")
        return output
