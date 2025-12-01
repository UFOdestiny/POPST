from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine


class DCRNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DCRNN_Engine, self).__init__(**args)

    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter)


class DCRNN_Engine_Quantile(CQR_Engine):
    def __init__(self, **args):
        super(DCRNN_Engine_Quantile, self).__init__(**args)

    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter)
