# import torch
# import torch.nn as nn
# from base.model import BaseModel
# from statsmodels.tsa.arima.model import ARIMA as A
# import warnings
# from concurrent.futures import ThreadPoolExecutor, as_completed
# warnings.filterwarnings("ignore")  # ARIMA 的未来警告略过

# class ARIMA(BaseModel):
#     def __init__(self, node_num, input_dim, output_dim, arima_order, **args):
#         super(ARIMA, self).__init__(node_num, input_dim, output_dim)
#         self.arima_order = arima_order
#         self.max_workers = 8

#     def _fit_single_series(self, series_info):
#         """
#         Fit ARIMA to a single time series and forecast.
#         series_info: (b, i, j, series)
#         """
#         b, i, j, series = series_info
#         print(b, i, j)
#         try:
#             model = ARIMA(series, order=self.order)
#             fitted = model.fit()
#             forecast = fitted.forecast(steps=1)
#             return (b, i, j, forecast[0])
#         except Exception:
#             return (b, i, j, series[-1])  # fallback to last value


#     def forward(self, input, label=None):  # (b, t, n, f)
#         # print(input.shape)
#         B, T, N, F = input.shape
#         output = torch.zeros(B, self.horizon, N, F)

#         # Prepare series list
#         series_list = [
#             (b, i, j, input[b, :, i, j])
#             for b in range(B)
#             for i in range(N)
#             for j in range(N)
#         ]

#         # Run multithreaded ARIMA fitting
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = [executor.submit(self._fit_single_series, series_info) for series_info in series_list]
#             for future in as_completed(futures):
#                 b, i, j, val = future.result()
#                 output[b, 0, i, j] = val
#         # print("output~")
#         return output


import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from base.model import BaseModel


class ARIMA_(BaseModel):
    def __init__(self, order=(1, 1, 0), n_threads=8, **args):
        super().__init__(**args)
        """
        ARIMA 多序列预测（多线程版）
        :param order: tuple (p,d,q)
        :param n_threads: 并行线程数
        """
        self.order = order
        self.n_threads = n_threads

    def _fit_forecast(self, ts, steps, idx):
        i, j = idx
        try:
            model = ARIMA(ts, order=self.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return (i, j, forecast)
        except Exception as e:
            print(f"ARIMA failed at ({i},{j}): {e}")
            return (i, j, np.full(steps, np.nan))

    def forward(self, X_train, Y_test_len):
        """
        拟合训练集并预测测试集长度
        :param X_train: numpy array, shape (T_train, N, N)
        :param Y_test_len: int, 测试集长度
        :return: Y_pred, shape (Y_test_len, N, N)
        """
        T_train, N, _ = X_train.shape
        Y_pred = np.zeros((Y_test_len, N, N))

        # 准备任务列表
        tasks = [((i, j), X_train[:, i, j]) for i in range(N) for j in range(N)]

        # 使用 ThreadPoolExecutor 多线程执行
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(self._fit_forecast, ts, Y_test_len, idx): idx
                for idx, ts in tasks
            }

            for f in futures:
                i, j, forecast = f.result()
                Y_pred[:, i, j] = forecast

            # # tqdm()
            # for f in tqdm(
            #     as_completed(futures), total=len(futures), desc="ARIMA predicting"
            # ):
            #     i, j, forecast = f.result()
            #     Y_pred[:, i, j] = forecast

        return Y_pred
