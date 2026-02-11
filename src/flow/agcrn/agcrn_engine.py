from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine


class AGCRN_Engine(BaseEngine):
    def __init__(self, **args):
        args["init_weights"] = True
        super().__init__(**args)


class AGCRN_Engine_Quantile(CQR_Engine):
    def __init__(self, **args):
        args["init_weights"] = True
        super().__init__(**args)
