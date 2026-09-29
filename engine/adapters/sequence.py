from engine.trainer import BaseEngine
from engine.calibration.quantile import CQR_Engine


class DCRNN_Engine(BaseEngine):
    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter)


class DCRNN_Engine_Quantile(CQR_Engine):
    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter)


class DGCRN_Engine(BaseEngine):
    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter, self.model.horizon)


class DGCRN_Engine_Quantile(CQR_Engine):
    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter, self.model.horizon)
