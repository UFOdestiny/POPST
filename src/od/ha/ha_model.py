import torch.nn as nn
import torch
from base.model import BaseModel
import numpy as np


class HA(BaseModel):
    def __init__(self, **args):
        super(HA, self).__init__(**args)

    def forward(self, X_train, Y_test_len):
        avg = np.mean(X_train, axis=0)
        return np.repeat(avg[np.newaxis, :, :], Y_test_len, axis=0)
