from base.engine import BaseEngine
from base.CQR_engine import CQR_Engine


class DGCRN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DGCRN_Engine, self).__init__(**args)
    
    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter, self.model.horizon)
    

class DGCRN_Engine_Quantile(CQR_Engine):
    def __init__(self, **args):
        super(DGCRN_Engine_Quantile, self).__init__(**args)
    
    def _predict(self, x, label, iter, *args):
        return self.model(x, label, iter, self.model.horizon)
    