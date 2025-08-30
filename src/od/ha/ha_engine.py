import torch
import numpy as np
from base.engine import BaseEngine
import time

class HA_Engine(BaseEngine):
    def __init__(self, **args):
        super(HA_Engine, self).__init__(**args)

    def train(self, export):
        train, valid, test = self._dataloader

        # train = train[:, :10, :10]
        # test = test[:, :10, :10]

        pred = self.model(train, test.shape[0])

        pred = torch.from_numpy(pred)
        test = torch.from_numpy(test)

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
