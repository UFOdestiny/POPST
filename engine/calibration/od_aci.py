"""Adaptive conformal inference for post-hoc OD prediction intervals.

The engine wraps an already-trained point or ZINB OD forecaster, exactly as
``OD_CQR_Engine`` does.  It never changes model weights or centre forecasts.
Validation residuals form a fixed reference distribution; while the ordered
test stream is evaluated, ACI causally updates the target miscoverage after
each forecast origin from the observed interval error.  This lets intervals
widen after undercoverage and narrow after overcoverage under distribution
shift.
"""

import numpy as np
import torch
from engine.calibration.od_split import OD_CQR_Engine


class OD_ACI_Engine(OD_CQR_Engine):
    """Post-hoc, causal Adaptive Conformal Inference for OD tensors.

    ``calibration.mode=horizon`` maintains one ACI controller per forecast horizon;
    ``calibration.mode=global`` maintains one controller across all horizons.  To avoid
    retaining every cell of a large OD validation set, each controller uses a
    deterministic, evenly-spaced residual reservoir of bounded size.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.gamma = float(getattr(self.config.calibration, "aci_gamma", 0.005))
        if self.gamma <= 0:
            raise ValueError("od_aci_gamma must be positive")
        self.calibration_size = max(
            int(getattr(self.config.calibration, "aci_calibration_size", 200000)), 1
        )
        self._aci_reference = None
        self._aci_alpha = None

    def _reservoir(self, scores):
        """Sorted finite residual reservoir, bounded for large OD matrices."""
        values = scores[torch.isfinite(scores)].detach().cpu().numpy().reshape(-1)
        if values.size == 0:
            return np.zeros(1, dtype=np.float32)
        if values.size > self.calibration_size:
            index = np.linspace(0, values.size - 1, self.calibration_size, dtype=np.int64)
            values = values[index]
        values.sort()
        return values

    @staticmethod
    def _reference_quantile(reference, alpha):
        alpha = float(np.clip(alpha, 1e-6, 1.0 - 1e-6))
        rank = min(int(np.ceil((reference.size + 1) * (1.0 - alpha))), reference.size) - 1
        return float(reference[rank])

    def calibrate(self, mode="val"):
        pred, label = self._collect(mode)
        scores = torch.abs(label - pred)
        if self.cqr_mode == "horizon":
            self._aci_reference = [self._reservoir(scores[:, h]) for h in range(scores.shape[1])]
            self._aci_alpha = np.full(scores.shape[1], self.alpha, dtype=np.float64)
            sizes = ", ".join(str(ref.size) for ref in self._aci_reference)
        else:
            self._aci_reference = [self._reservoir(scores)]
            self._aci_alpha = np.asarray([self.alpha], dtype=np.float64)
            sizes = str(self._aci_reference[0].size)
        self._logger.info(
            f"OD-ACI calibration on '{mode}' (target coverage={1 - self.alpha:.1%}, "
            f"mode={self.cqr_mode}, gamma={self.gamma:g}, reference sizes={sizes})"
        )

    def _interval_at_origin(self, pred_t):
        """Issue intervals for one forecast origin before consuming its label."""
        lower, upper = torch.empty_like(pred_t), torch.empty_like(pred_t)
        if self.cqr_mode == "horizon":
            for h, reference in enumerate(self._aci_reference):
                q = self._reference_quantile(reference, self._aci_alpha[h])
                lower[h], upper[h] = (pred_t[h] - q).clamp_min(0), pred_t[h] + q
        else:
            q = self._reference_quantile(self._aci_reference[0], self._aci_alpha[0])
            lower.copy_((pred_t - q).clamp_min(0))
            upper.copy_(pred_t + q)
        return lower, upper


    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode != "test":
            return super().evaluate(mode, model_path, export, train_test)
        if not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            self.calibrate()
        pred, label = self._collect("test")
        if self._aci_reference is None:
            self.calibrate()

        lower, upper = torch.empty_like(pred), torch.empty_like(pred)
        # A horizon-h target issued at origin i arrives at i+h+1. Consume it
        # only once that timestamp has closed; this also handles gaps in origins.
        indices = self._dataloader["test_loader"].dataset.indices
        pending = np.zeros(pred.shape[1], dtype=np.int64)
        for t in range(pred.shape[0]):
            for h in range(pred.shape[1]):
                while pending[h] < t and indices[pending[h]] + h + 1 <= indices[t]:
                    k = pending[h]
                    valid = torch.isfinite(label[k, h])
                    if valid.any():
                        missed = ((label[k, h] < lower[k, h]) | (label[k, h] > upper[k, h]))[valid].float().mean()
                        controller = h if self.cqr_mode == "horizon" else 0
                        self._aci_alpha[controller] = np.clip(
                            self._aci_alpha[controller] + self.gamma * (self.alpha - float(missed)),
                            1e-6, 1.0 - 1e-6,
                        )
                    pending[h] += 1
            lower[t], upper[t] = self._interval_at_origin(pred[t])

        metric_pred, metric_lower, metric_upper, metric_label = (
            pred.to(self._device),
            lower.to(self._device),
            upper.to(self._device),
            label.to(self._device),
        )
        mask_value = torch.tensor(float("nan"))
        for h in range(self.model.horizon):
            self.metric.compute_one_batch(
                self._horizon_slice(metric_pred, h),
                self._horizon_slice(metric_label, h),
                mask_value,
                "test",
                lower=self._horizon_slice(metric_lower, h),
                upper=self._horizon_slice(metric_upper, h),
                alpha=self.alpha,
            )
        if not train_test:
            with self._logger.no_time():
                self._logger.info("\n" + "=" * 25 + "     Test (OD-ACI)     " + "=" * 25)
            for msg in self.metric.get_test_msg():
                self._logger.info(msg)
            self._logger.info(
                "OD-ACI final alpha: " + ", ".join(f"{alpha:.5f}" for alpha in self._aci_alpha)
            )
        if export:
            result = torch.stack([pred, lower, upper, label]).cpu().numpy()
            tag = getattr(self.config.calibration, "tag", "").strip()
            suffix = f"od_aci_{tag}_res" if tag else "od_aci_res"
            path = self._get_unique_save_path(suffix)
            np.save(path, result)
            self._logger.info(f"OD-ACI Results Save Path: {path} | Shape: {result.shape}")
