"""Ordinary OD engine with MSE/MAE plus OD-marginal regularization."""

from base.engine import BaseEngine_OD
from base.metrics import masked_mae, masked_mse, register_metric
from src.od.pdr_v2.pdr_v2_engine import production_attraction_loss


def mse_od(preds, labels, null_val, lambda_pa=0.1):
    """Masked MSE plus weighted production--attraction consistency loss."""
    return masked_mse(preds, labels, null_val) + lambda_pa * production_attraction_loss(
        preds, labels, null_val
    )


def mae_od(preds, labels, null_val, lambda_pa=0.1):
    """Masked MAE plus weighted production--attraction consistency loss."""
    return masked_mae(preds, labels, null_val) + lambda_pa * production_attraction_loss(
        preds, labels, null_val
    )


def register_od_losses(lambda_pa=0.1):
    """Register ``MSE_OD`` and ``MAE_OD`` with the requested PA weight."""
    if lambda_pa < 0:
        raise ValueError(f"lambda_pa must be non-negative, got {lambda_pa}")
    weight = float(lambda_pa)
    register_metric(
        "MSE_OD",
        lambda preds, labels, null_val: mse_od(preds, labels, null_val, weight),
        "basic",
    )
    register_metric(
        "MAE_OD",
        lambda preds, labels, null_val: mae_od(preds, labels, null_val, weight),
        "basic",
    )


# Make both names available as soon as this module is imported.  The engine
# registers them again before Metrics is constructed if --lambda_pa overrides
# the default value.
register_od_losses()


class PDRRegLossEngine(BaseEngine_OD):
    """BaseEngine_OD with configurable registrations for the two OD losses.

    No training/evaluation method is overridden: this remains the standard OD
    engine, with only its loss registry entries configured before Metrics is
    initialized.
    """

    def __init__(self, *args, lambda_pa=0.1, **kwargs):
        register_od_losses(lambda_pa)
        super().__init__(*args, **kwargs)
        self.lambda_pa = float(lambda_pa)
        self._logger.info(f"{'PA Loss Weight':20s}: {self.lambda_pa}")
