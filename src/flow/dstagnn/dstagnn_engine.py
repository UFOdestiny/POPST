import torch
from base.engine import BaseEngine
from base.metrics import Metrics
from base.CQR_engine import CQR_Engine


class DSTAGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DSTAGNN_Engine, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

class DSTAGNN_Engine_Quantile(CQR_Engine):
    def __init__(self, **args):
        super(DSTAGNN_Engine_Quantile, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)