"""PDR engine with production--attraction consistency regularization."""

import torch

from base.metrics import get_mask
from src.od.pdr.pdr_engine import PDR_Engine, zinb_mean, zinb_nll


def production_attraction_loss(pred, target, null_val=float("nan")):
    """Compare predicted and observed OD marginals in ``log1p`` space.

    Both tensors follow the framework's ``(B, H, O, D, C)`` OD layout.  For
    missing cells, the corresponding prediction and target are omitted from
    the marginal on both sides.  The two terms are the mean absolute errors
    of production (row sums) and attraction (column sums), respectively.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} and target {target.shape} must match")
    if pred.ndim != 5:
        raise ValueError(
            "production_attraction_loss expects OD tensors shaped "
            f"(B, H, O, D, C), got {pred.shape}"
        )

    valid = get_mask(target, null_val).bool()
    pred = torch.where(valid, pred.clamp_min(0.0), torch.zeros_like(pred))
    target = torch.where(valid, target.clamp_min(0.0), torch.zeros_like(target))

    pred_production = pred.sum(dim=3)
    true_production = target.sum(dim=3)
    production_valid = valid.any(dim=3)

    pred_attraction = pred.sum(dim=2)
    true_attraction = target.sum(dim=2)
    attraction_valid = valid.any(dim=2)

    production_error = torch.abs(
        torch.log1p(pred_production) - torch.log1p(true_production)
    )
    attraction_error = torch.abs(
        torch.log1p(pred_attraction) - torch.log1p(true_attraction)
    )

    production_loss = (production_error * production_valid).sum()
    production_loss = production_loss / production_valid.sum().clamp_min(1)
    attraction_loss = (attraction_error * attraction_valid).sum()
    attraction_loss = attraction_loss / attraction_valid.sum().clamp_min(1)
    return production_loss + attraction_loss


class PDR_v2_Engine(PDR_Engine):
    """PDR engine augmented with the small PA loss described in design.tex.

    Validation, testing, checkpoint selection, metrics, and prediction remain
    inherited from :class:`PDR_Engine`; only the training objective changes to
    ``ZINB NLL + lambda_pa * PA``.
    """

    def __init__(self, *args, lambda_pa=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        if lambda_pa < 0:
            raise ValueError(f"lambda_pa must be non-negative, got {lambda_pa}")
        self.lambda_pa = float(lambda_pa)
        self._logger.info(f"{'PA Loss Weight':20s}: {self.lambda_pa}")

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n"
                    + "=" * 25 + "   Training   " + "=" * 25
                )
            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])

            n, p, pi = self.model(X)
            label_c = self._to_counts(label)
            nll = zinb_nll(label_c, n, p, pi, null_val=mask_value)
            pred = zinb_mean(n, p, pi)
            pa_loss = production_attraction_loss(pred, label_c, null_val=mask_value)
            loss = nll + self.lambda_pa * pa_loss

            # Preserve PDR's metric semantics: NLL is the reported training
            # objective and remains the validation/checkpoint-selection metric.
            with torch.no_grad():
                self.metric.compute_one_batch(
                    pred, label_c, mask_value, "train", value=nll
                )

            loss.backward()
            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1
