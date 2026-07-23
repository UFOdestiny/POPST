"""Likelihood engine shared by PDR probabilistic-regression baselines."""

import math

import torch

from base.engine import BaseEngine_OD
from base.metrics import get_mask


def distribution_nll(
    y,
    loc,
    scale,
    distribution,
    null_val=float("nan"),
    student_df=3.0,
):
    """Masked mean NLL for a univariate location-scale distribution."""
    if y.shape != loc.shape or y.shape != scale.shape:
        raise ValueError(
            f"Distribution parameter shapes must match labels: "
            f"y={y.shape}, loc={loc.shape}, scale={scale.shape}"
        )

    mask = get_mask(y, null_val)
    # Replace masked NaNs before arithmetic.  Masking only after computing the
    # likelihood can still leave NaN derivatives in autograd (0 * NaN).
    safe_y = torch.where(mask.bool(), y, loc.detach())
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    residual = (safe_y - loc) / scale

    if distribution == "gaussian":
        nll = (
            0.5 * residual.square()
            + torch.log(scale)
            + 0.5 * math.log(2.0 * math.pi)
        )
    elif distribution == "laplace":
        nll = residual.abs() + torch.log(scale) + math.log(2.0)
    elif distribution == "student_t":
        if student_df <= 1.0:
            raise ValueError("student_df must be greater than 1 so the mean exists")
        df = y.new_tensor(float(student_df))
        nll = (
            torch.lgamma(df / 2.0)
            - torch.lgamma((df + 1.0) / 2.0)
            + 0.5 * (torch.log(df) + math.log(math.pi))
            + torch.log(scale)
            + 0.5 * (df + 1.0) * torch.log1p(residual.square() / df)
        )
    else:
        raise ValueError(f"Unsupported regression distribution: {distribution}")

    count = mask.sum()
    if count == 0:
        return (loc.sum() + scale.sum()) * 0.0
    nll = torch.where(mask.bool(), nll, torch.zeros_like(nll))
    nll = torch.nan_to_num(nll, nan=0.0, posinf=1e12, neginf=-1e12)
    return nll.sum() / count


class PDRRegDistributionEngine(BaseEngine_OD):
    """Train a PDR location-scale model with a distribution-specific NLL.

    The likelihood is evaluated in the same space as the model output
    (normally the standardized data space).  Point metrics use the location,
    inverse-transformed to the original OD-demand space.
    """

    distribution = None

    def __init__(self, student_df=3.0, **kwargs):
        super().__init__(**kwargs)
        self.student_df = float(student_df)
        if self.distribution == "student_t" and self.student_df <= 1.0:
            raise ValueError("student_df must be greater than 1 so the mean exists")

    def _nll(self, label, loc, scale, null_val):
        return distribution_nll(
            label,
            loc,
            scale,
            self.distribution,
            null_val=null_val,
            student_df=self.student_df,
        )

    def _point_space(self, loc, label):
        if self._normalize:
            loc, label = self._inverse_transform(
                [loc, label], device=self._device.type
            )
        return loc, label

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n"
                    + "=" * 25
                    + "   Training   "
                    + "=" * 25
                )

            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])
            loc, scale = self.model(X)
            loss = self._nll(label, loc, scale, mask_value)

            with torch.no_grad():
                pred, point_label = self._point_space(loc, label)
                self.metric.compute_one_batch(
                    pred,
                    point_label,
                    mask_value,
                    "train",
                    value=loss.detach(),
                )

            if not torch.isfinite(loss):
                self._optimizer.zero_grad()
                self._iter_cnt += 1
                continue

            loss.backward()
            if self._clip_grad_norm != 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
                if not torch.isfinite(total_norm):
                    self._optimizer.zero_grad()
                    self._iter_cnt += 1
                    continue
            self._optimizer.step()
            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        preds, labels = [], []
        likelihood_labels, locs, scales = [], [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                loc, scale = self.model(X)
                pred, point_label = self._point_space(loc, label)

                if mode == "val":
                    nll = self._nll(
                        label, loc, scale, self._mask_value.to(loc.device)
                    )
                    self.metric.compute_one_batch(
                        pred,
                        point_label,
                        self._mask_value.to(pred.device),
                        "valid",
                        value=nll,
                    )
                else:
                    preds.append(self._collect(pred).cpu())
                    labels.append(self._collect(point_label).cpu())
                    likelihood_labels.append(self._collect(label).cpu())
                    locs.append(self._collect(loc).cpu())
                    scales.append(self._collect(scale).cpu())

        if mode == "val":
            return

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        likelihood_labels = torch.cat(likelihood_labels, dim=0)
        locs = torch.cat(locs, dim=0)
        scales = torch.cat(scales, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                nll = self._nll(
                    self._horizon_slice(likelihood_labels, i),
                    self._horizon_slice(locs, i),
                    self._horizon_slice(scales, i),
                    mask_value,
                )
                self.metric.compute_one_batch(
                    self._horizon_slice(preds, i),
                    self._horizon_slice(labels, i),
                    mask_value,
                    "test",
                    value=nll,
                )

            if not train_test:
                with self._logger.no_time():
                    self._logger.info(
                        "\n" + "=" * 25 + "     Test     " + "=" * 25
                    )
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(preds, labels)
                self.save_test()


class PDRRegGaussianEngine(PDRRegDistributionEngine):
    distribution = "gaussian"


class PDRRegLaplaceEngine(PDRRegDistributionEngine):
    distribution = "laplace"


class PDRRegStudentTEngine(PDRRegDistributionEngine):
    distribution = "student_t"
