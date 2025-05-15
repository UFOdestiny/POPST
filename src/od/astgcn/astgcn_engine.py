import torch
from base.engine import BaseEngine
from base.quantile_engine import Quantile_Engine


class ASTGCN_Engine(BaseEngine):
    def __init__(self, **args):
        super(ASTGCN_Engine, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

class ASTGCN_Engine_Quantile(Quantile_Engine):
    def __init__(self, **args):
        super(ASTGCN_Engine_Quantile, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)