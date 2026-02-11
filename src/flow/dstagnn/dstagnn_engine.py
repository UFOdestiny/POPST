from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine


class DSTAGNN_Engine(BaseEngine):
    def __init__(self, **args):
        args["init_weights"] = True
        super().__init__(**args)


class DSTAGNN_Engine_Quantile(CQR_Engine):
    def __init__(self, **args):
        args["init_weights"] = True
        super().__init__(**args)
