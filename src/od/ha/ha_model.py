import torch.nn as nn
import torch
from base.model import BaseModel
import numpy as np


class HA(BaseModel):
    def __init__(self, step=6, **args):
        super(HA, self).__init__(**args)
        self.step = step

    def forward(self, valid, test):
        temp = valid[-6:]
        his = np.concat([temp, test],axis=0)
        preds = []

        for i in range(test.shape[0]):
            window = np.array(his[i : i + self.step])
            avg = np.mean(window, axis=0)
            preds.append(avg)

        return np.array(preds)
