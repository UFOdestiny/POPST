import torch
import numpy as np
from base.engine import BaseEngine
import time

from statsmodels.tsa.arima.model import ARIMA as A
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")  # ARIMA 的未来警告略过


class ARIMA_Engine(BaseEngine):
    def __init__(self, **args):
        super(ARIMA_Engine, self).__init__(**args)

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        test_x = []
        test_y = []
        for X, label in self._dataloader["test_loader"].get_iterator():
            X, label = self._to_device(self._to_tensor([X, label]))
            test_x.append(X)
            test_y.append(label)
        test_x = torch.cat(test_x, dim=0)
        test_y = torch.cat(test_y, dim=0)

        test_x = test_x.detach().cpu().numpy() #[:1, :, :5, :5]
        test_y = test_y.detach().cpu().numpy() #[:1, :, :5, :5]

        def _fit(series_info):
            b, f, to, T = series_info
            # print(f"batch: {b},{f},{to},{T}\n")
            model = A(T, order=(2, 1, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)

            return (b, f, to, forecast[0])

        B, T, F, To = test_x.shape
        output = torch.zeros(B, 1, F, To)

        series_list = [
            (b, f, t, test_x[b, :, f, t])
            for b in range(B)
            for f in range(F)
            for t in range(To)
        ]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_fit, series_info) for series_info in series_list
            ]
            for future in as_completed(futures):
                b, f, to, val = future.result()
                output[b, 0, f, to] = val

        self.metric.compute_one_batch(
            torch.tensor(output),
            torch.tensor(test_y),
            torch.tensor(0),
            "test",
            scale=None,
        )

        for i in self.metric.get_test_msg():
            self._logger.info(i)

        if export:
            self.save_result(torch.tensor(output), torch.tensor(test_y))
