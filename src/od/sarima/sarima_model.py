import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from base.model import BaseModel
from sklearn.decomposition import TruncatedSVD
from statsmodels.tsa.statespace.varmax import VARMAX
import warnings

warnings.filterwarnings("ignore")  # ARIMA 的未来警告略过


class SARIMA_(BaseModel):
    def __init__(self, order=(6, 0, 0), n_threads=8, **args):
        super().__init__(**args)
        """
        :param order: tuple (p,d,q)
        :param n_threads: 并行线程数
        """
        self.order = order
        self.n_threads = n_threads

    def _fit_forecast(self, ts, steps, idx):
        i, j = idx
        try:
            model = SARIMA(ts, order=self.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return (i, j, forecast)
        except Exception as e:
            print(f"ARIMA failed at ({i},{j}): {e}")
            return (i, j, np.full(steps, np.nan))

    def forward(self, X_train, Y_test_len):
        """
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

            # for f in futures:
            #     i, j, forecast = f.result()
            #     Y_pred[:, i, j] = forecast

            # tqdm()
            for f in tqdm(
                as_completed(futures), total=len(futures), desc="ARIMA predicting"
            ):
                i, j, forecast = f.result()
                Y_pred[:, i, j] = forecast

        return Y_pred
