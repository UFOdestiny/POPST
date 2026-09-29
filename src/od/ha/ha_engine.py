from base.engine import BaseEngine_OD_Stat


class HA_Engine(BaseEngine_OD_Stat):
    """Historical Average engine: forecasts each test step from the rolling mean
    of preceding observations (seeded with the validation tail), rather than the
    fit-on-train protocol of ARIMA/VAR."""

    def _forecast(self, train, valid, test):
        return self.model(valid, test)
