import os
import numpy as np
import sys
import torch

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
sys.path.append("/home/dy23a.fsu/st/")

from base.engine import BaseEngine
import time

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")  # ARIMA 的未来警告略过


class SARIMA_Engine(BaseEngine):
    def __init__(self, **args):
        super(SARIMA_Engine, self).__init__(**args)

    def train(self, export):
        train, valid, test = self._dataloader

        # train = train[:, :10, :10]
        # test = test[:, :10, :10]

        pred = self.model(train, test.shape[0])

        pred = torch.from_numpy(pred)
        test = torch.from_numpy(test)

        if self._normalize:
            pred, test = self._inverse_transform([pred, test], device="cpu")

        self.metric.compute_one_batch(
            pred,
            test,
            torch.tensor(0),
            "test",
        )

        for i in self.metric.get_test_msg():
            self._logger.info(i)

        if export:
            self.save_result(pred, test)
