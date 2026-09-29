import torch
import numpy as np
from engine.trainer import BaseEngine_OD


class RollingStatisticalEngine(BaseEngine_OD):
    """Same origin and target windows as neural baselines, in original counts."""

    def train(self, export=False):
        data = self._dataloader
        origins = data["origins"]
        horizon = self.model.horizon
        pred = self.model.forecast_origins(data["series"], data["train_end"], origins, horizon)
        target = data["series"][origins[:, None] + np.arange(1, horizon + 1)]
        if pred.shape != target.shape or not np.isfinite(pred).all():
            raise RuntimeError("Invalid statistical forecast output")
        pred, target = torch.from_numpy(pred).float(), torch.from_numpy(target.copy()).float()
        for h in range(horizon):
            self.metric.compute_one_batch(pred[:, h:h+1], target[:, h:h+1], float("nan"), "test")
        for message in self.metric.get_test_msg():
            self._logger.info(message)
        if export:
            self.save_result(pred, target)
