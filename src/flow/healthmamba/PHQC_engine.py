"""PHQC — Post-Hoc Quantile Calibration engine (HealthMamba).

This engine implements the *comprehensive uncertainty quantification* and
*post-hoc quantile calibration* described in HealthMamba (IJCAI'26):
"HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model
for Effective and Reliable Healthcare Facility Visit Prediction".

Unlike :class:`base.CQR_engine.CQR_Engine` (split conformal on a single
quantile head), HealthMamba trains *three complementary* uncertainty
mechanisms jointly and then calibrates the resulting interval once on a
held-out split:

1. **Node-based** (quantile regression).  The model emits ordered lower /
   median / upper quantiles per node-time-feature; trained with the
   pinball loss ``L_quant``.

2. **Distribution-based** (heteroscedastic Gaussian).  A second head emits
   ``(mu, sigma^2)`` per node-time-feature; trained with the Gaussian NLL
   ``L_nll`` and a calibration loss ``L_calib`` that forces the
   standardized residuals ``r = (y - mu) / sigma`` to zero mean / unit
   variance.

3. **Parameter-based** (MC dropout).  At inference ``M`` stochastic forward
   passes decompose the predictive variance into aleatoric + epistemic
   parts (Eq. 18-19).  The epistemic part widens the quantile interval.
   A consistency penalty ``L_param`` (the variance of the stochastic mean
   head, returned by the model as ``mc_var``) is minimised during training.

The integrated objective is::

    L_total = L_quant + w_nll * L_nll + w_param * L_param + w_calib * L_calib

**Post-hoc quantile calibration** (Sec. 4.3.5).  After training, on a
held-out calibration split we compute the one-sided adjustment margin::

    c = Quantile_{1-alpha} { max(l_i - y_i, y_i - u_i, 0) }

and widen every test interval to ``[l - c, u + c]``.  The ``max(., 0)``
clamp makes ``c >= 0`` — calibration only *widens* intervals to reach the
target coverage (it never shrinks below the learned quantiles).  ``c`` is
per-horizon by default (``--phqc horizon``) or scalar (``--phqc global``),
computed at the conformal level ``ceil((n+1)(1-alpha))/n`` for the
finite-sample coverage guarantee, and persisted in the checkpoint.

The model's ``forward`` returns a dict with normalised-space tensors
``q_lo, q_mid, q_hi, mu, logvar`` (each ``(B, H, N, F)``) and a scalar-ish
``mc_var`` (same shape) used only by ``L_param``.  All calibration and
metrics run in the original (inverse-transformed) data space; the
per-feature inverse transform is monotonic, so quantile ordering is kept.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from base.engine import BaseEngine
from base.metrics import Metrics


class PHQC_Engine(BaseEngine):
    DEFAULT_METRICS = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "IS", "COV"]

    def __init__(
        self,
        mc_samples: int = 20,
        w_nll: float = 1.0,
        w_param: float = 1.0,
        w_calib: float = 1.0,
        nll_eps: float = 1e-6,
        **args,
    ):
        """
        Args:
            mc_samples: ``M`` — number of MC-dropout forward passes used at
                inference for the parameter-based (epistemic) uncertainty.
            w_nll / w_param / w_calib: weights of the distribution-based,
                parameter-based, and calibration loss terms (paper uses an
                unweighted sum, i.e. all 1.0).
            nll_eps: numerical-stability floor for the Gaussian variance.
        """
        args["loss_fn"] = "Quantile"
        args["metric_list"] = self.DEFAULT_METRICS
        super().__init__(**args)

        self.mc_samples = int(getattr(self.args, "phqc_mc", mc_samples))
        self.w_nll = float(getattr(self.args, "phqc_w_nll", w_nll))
        self.w_param = float(getattr(self.args, "phqc_w_param", w_param))
        self.w_calib = float(getattr(self.args, "phqc_w_calib", w_calib))
        self.nll_eps = nll_eps

        # Target miscoverage and the matching quantile levels.
        self.alpha = float(getattr(self.args, "quantile_alpha", 0.1))
        self.q_lower = self.alpha / 2.0
        self.q_upper = 1.0 - self.alpha / 2.0
        # Normal quantile z_{1-alpha/2} used to widen by the epistemic std.
        self.z = float(_norm_ppf(1.0 - self.alpha / 2.0))

        # Post-hoc calibration granularity: "horizon" (default) or "global".
        self.phqc_mode = getattr(self.args, "phqc", "horizon")

        # Adjustment margin c; filled by calibrate(), persisted in checkpoints.
        self._margin = None

        self.metric = Metrics(
            self._loss_fn, args["metric_list"], horizon=self.model.horizon
        )

    # -- quantile levels forwarded to the interval metrics ------------------

    def _quantile_kw(self):
        return {"q_lower": self.q_lower, "q_upper": self.q_upper, "alpha": self.alpha}

    # -- model output handling ---------------------------------------------

    def _as_dict(self, out):
        """Normalise model output to a dict (a bare tensor is treated as the
        median, with degenerate quantiles / Gaussian)."""
        if isinstance(out, dict):
            return out
        if isinstance(out, tuple):
            out = out[0]
        return {"q_lo": out, "q_mid": out, "q_hi": out, "mu": out,
                "logvar": torch.zeros_like(out), "mc_var": torch.zeros_like(out)}

    # -- losses -------------------------------------------------------------

    def _nll_loss(self, mu, logvar, label):
        """Gaussian negative log-likelihood (normalised space), NaN-masked."""
        mask = (~torch.isnan(label)).float()
        var = F.softplus(logvar) + self.nll_eps
        y = torch.nan_to_num(label, nan=0.0)
        loss = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom

    def _calib_loss(self, mu, logvar, label):
        """Standardized-residual calibration loss (Eq. 17): zero mean, unit var."""
        mask = (~torch.isnan(label)).float()
        std = torch.sqrt(F.softplus(logvar) + self.nll_eps)
        y = torch.nan_to_num(label, nan=0.0)
        r = (y - mu) / (std + self.nll_eps)
        denom = mask.sum().clamp(min=1.0)
        r = r * mask
        mean_r = r.sum() / denom
        mean_r2 = (r ** 2).sum() / denom
        return mean_r ** 2 + (mean_r2 - 1.0) ** 2

    def _param_loss(self, mc_var, label):
        """Parameter-based consistency penalty: minimise epistemic variance."""
        mask = (~torch.isnan(label)).float()
        denom = mask.sum().clamp(min=1.0)
        return (mc_var * mask).sum() / denom

    # -- training -----------------------------------------------------------

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

            out = self._as_dict(self.model(X))
            q_lo, q_mid, q_hi = out["q_lo"], out["q_mid"], out["q_hi"]

            # Pinball loss is computed in the original data space (like CQR):
            # inverse-transform the quantiles and the label, then let the
            # metric tracker both log and return the gradient term.
            if self._normalize:
                lo_o, mid_o, hi_o, label_o = self._inverse_transform(
                    [q_lo, q_mid, q_hi, label], device=self._device.type
                )
            else:
                lo_o, mid_o, hi_o, label_o = q_lo, q_mid, q_hi, label

            l_quant = self.metric.compute_one_batch(
                mid_o, label_o, mask_value, "train",
                lower=lo_o, upper=hi_o, **self._quantile_kw(),
            )

            # Distribution / parameter / calibration terms in normalised space.
            l_nll = self._nll_loss(out["mu"], out["logvar"], label)
            l_calib = self._calib_loss(out["mu"], out["logvar"], label)
            l_param = self._param_loss(out["mc_var"], label)

            total = (l_quant
                     + self.w_nll * l_nll
                     + self.w_param * l_param
                     + self.w_calib * l_calib)
            total.backward()

            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1

    # -- inference: MC-dropout intervals -----------------------------------

    def _single_intervals(self, X):
        """One forward pass -> normalised (lower, mid, upper)."""
        out = self._as_dict(self.model(X))
        return out["q_lo"], out["q_mid"], out["q_hi"]

    def _mc_intervals(self, X):
        """MC-dropout inference (Eq. 18-19).

        Runs ``M`` stochastic passes with dropout enabled, averages the
        quantile heads, and widens the interval by the epistemic std
        (variance of the stochastic mean across passes).  Returns
        normalised-space ``(lower, mid, upper)``.
        """
        M = max(1, self.mc_samples)
        if M == 1:
            return self._single_intervals(X)

        was_training = self.model.training
        self.model.train()  # enable dropout; model uses LayerNorm/RMSNorm only
        los, mids, his, mus = [], [], [], []
        with torch.no_grad():
            for _ in range(M):
                out = self._as_dict(self.model(X))
                los.append(out["q_lo"])
                mids.append(out["q_mid"])
                his.append(out["q_hi"])
                mus.append(out["mu"])
        if not was_training:
            self.model.eval()

        lo = torch.stack(los).mean(0)
        mid = torch.stack(mids).mean(0)
        hi = torch.stack(his).mean(0)
        epi_std = torch.stack(mus).var(0, unbiased=False).clamp(min=0.0).sqrt()

        lo = lo - self.z * epi_std
        hi = hi + self.z * epi_std
        return lo, mid, hi

    def _forward_intervals(self, X, label, mc):
        lo, mid, hi = self._mc_intervals(X) if mc else self._single_intervals(X)
        if self._normalize:
            lo, mid, hi, label = self._inverse_transform(
                [lo, mid, hi, label], device=self._device.type
            )
        return lo, mid, hi, label

    def _collect(self, mode, mc):
        """Run the model over a split; return stacked (lower, mid, upper,
        label) on CPU (original data space)."""
        self.model.eval()
        los, mids, his, labels = [], [], [], []
        loader_key = "test_loader" if mode == "export" else f"{mode}_loader"
        with torch.no_grad():
            for X, label in self._dataloader[loader_key].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                lo, mid, hi, label = self._forward_intervals(X, label, mc)
                los.append(lo.cpu())
                mids.append(mid.cpu())
                his.append(hi.cpu())
                labels.append(label.cpu())
        return (torch.cat(los), torch.cat(mids), torch.cat(his), torch.cat(labels))

    # -- post-hoc quantile calibration -------------------------------------

    @staticmethod
    def _calib_margin(scores, alpha):
        """``ceil((n+1)(1-alpha))/n`` empirical quantile of the one-sided
        nonconformity scores (NaN-safe)."""
        scores = scores[~torch.isnan(scores)]
        n = scores.numel()
        if n == 0:
            return torch.tensor(0.0)
        level = float(min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0))
        return torch.quantile(scores, level)

    def calibrate(self, mode="val"):
        """Compute the post-hoc adjustment margin ``c`` on a held-out split.

        Uses the same MC-dropout interval procedure as test so the margin
        matches the intervals it will correct.
        """
        lower, _, upper, label = self._collect(mode, mc=True)
        # One-sided nonconformity: max(l - y, y - u, 0) >= 0.
        scores = torch.clamp(torch.maximum(lower - label, label - upper), min=0.0)

        if self.phqc_mode == "global":
            c = self._calib_margin(scores, self.alpha)
            self._margin = c
            c_msg = f"{float(c):.4f}"
        else:
            T = scores.shape[1]
            c = torch.stack(
                [self._calib_margin(scores[:, h], self.alpha) for h in range(T)]
            )
            self._margin = c
            c_msg = ", ".join(f"h{h + 1}={v:.4f}" for h, v in enumerate(c.tolist()))

        n = int((~torch.isnan(scores)).sum().item())
        self._logger.info(
            f"PHQC calibration on '{mode}' (alpha={self.alpha}, mode={self.phqc_mode}, "
            f"M={self.mc_samples}, n={n}): c = {c_msg}"
        )
        return self._margin

    def _apply_margin(self, lower, upper):
        if self._margin is None:
            return lower, upper
        c = self._margin
        if torch.is_tensor(c) and c.ndim == 1:
            c = c.view(1, -1, *([1] * (lower.ndim - 2)))
        else:
            c = float(c)
        return lower - c, upper + c

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            if self._margin is None:
                self.calibrate()

        self.model.eval()

        # During training-time validation use a cheap single pass; the MC walk
        # and calibration are reserved for the test stream.
        if mode == "val":
            lower, mid, upper, label = self._collect("val", mc=False)
            self._compute_per_horizon(mid, lower, upper, label, "valid")
            return

        lower, mid, upper, label = self._collect(mode, mc=True)
        lower, upper = self._apply_margin(lower, upper)
        # Visit counts are non-negative; clamp the interval after calibration.
        lower = lower.clamp(min=0.0)
        mid = mid.clamp(min=0.0)
        upper = upper.clamp(min=0.0)

        if mode in {"test", "export"}:
            self._compute_per_horizon(mid, lower, upper, label, "test")

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(mid, lower, upper, label)

    def _compute_per_horizon(self, mids, lowers, uppers, labels, mode_name):
        mask_value = torch.tensor(torch.nan)
        for h in range(mids.shape[1]):
            self.metric.compute_one_batch(
                mids[:, h : h + 1], labels[:, h : h + 1], mask_value, mode_name,
                lower=lowers[:, h : h + 1], upper=uppers[:, h : h + 1],
                **self._quantile_kw(),
            )

    # -- export -------------------------------------------------------------

    def save_result(self, mids, lowers, uppers, labels):
        result = torch.stack([lowers, mids, uppers, labels], dim=0)
        base_name = f"{self.args.model_name}-{self.args.dataset}-res"
        save_name = f"{base_name}.npy"
        path = os.path.join(self._save_path, save_name)
        suffix = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{suffix}.npy"
            path = os.path.join(self._save_path, save_name)
            suffix += 1
        np.save(path, result.numpy())
        self._logger.info(
            f"Results Save Path: {path} | Shape: {result.shape} "
            f"(component, batch, horizon, node, feature)"
        )

    # -- checkpoint ---------------------------------------------------------

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        m = self._margin
        state = {
            "model": self.model.state_dict(),
            "margin": m.cpu() if torch.is_tensor(m) else m,
            "phqc_mode": self.phqc_mode,
            "alpha": self.alpha,
        }
        torch.save(state, os.path.join(save_path, self._time_model))

    def _load_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
            if checkpoint.get("margin") is not None:
                self._margin = checkpoint["margin"]
        else:
            self.model.load_state_dict(checkpoint, strict=False)

    def load_model(self, save_path):
        filename = self._time_model
        f = os.path.join(save_path, filename)
        if not os.path.exists(f):
            models = [i for i in os.listdir(save_path) if i.endswith(".pt")]
            if not models:
                self._logger.info(f"Model {f} Not Exist. No More Models.")
                exit()
            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            f = os.path.join(save_path, models[-1])
            self._logger.info(
                f"Model {filename} Not Exist. Try the Newest Model {os.path.basename(f)}."
            )
        checkpoint = torch.load(f, weights_only=False, map_location=self._device)
        self._load_state_dict(checkpoint)

    def load_exact_model(self, path):
        checkpoint = torch.load(path, weights_only=False, map_location=self._device)
        self._load_state_dict(checkpoint)


def _norm_ppf(p):
    """Standard-normal inverse CDF (Acklam's rational approximation).

    Avoids a hard scipy dependency; accurate to ~1e-9 over (0, 1).
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
