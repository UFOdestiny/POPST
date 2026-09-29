from base.model import BaseModel
import numpy as np


class HA(BaseModel):
    """Historical Average: predict each test step as the mean of the preceding
    ``step`` observations.  Shape-agnostic — operates on whole ``(T, N, N, D)``
    arrays, so all mobility channels are handled at once."""

    def __init__(self, step=6, **args):
        super(HA, self).__init__(**args)
        self.step = step

    def forward(self, valid, test):
        # Seed the rolling window with the tail of the validation split so the
        # first test steps have a full look-back.
        temp = valid[-self.step :]
        his = np.concatenate([temp, test], axis=0)
        preds = []
        for i in range(test.shape[0]):
            window = his[i : i + self.step]
            preds.append(np.mean(window, axis=0))
        return np.array(preds)
