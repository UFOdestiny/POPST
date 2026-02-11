import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from base.engine import BaseEngine


class VAR_Engine(BaseEngine):
    def __init__(self, **args):
        super(VAR_Engine, self).__init__(**args)

    def train(self, export):
        train, valid, test = self._dataloader

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
