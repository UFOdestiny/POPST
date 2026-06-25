"""Adaptive Sequential Conformalized Quantile Regression (AS-CQR).

This engine implements the calibration module of EnergyMamba
(arXiv:2606.00506v1).  It extends :class:`base.CQR_engine.CQR_Engine`
and reuses its whole training / quantile-parsing / checkpoint machinery;
only the *calibration and interval construction* are replaced.

Three changes versus vanilla CQR (Romano et al., 2019):

1. **Locally adaptive nonconformity** (Eq. 21).  The conformity score is
   normalised by the predicted interval width, making it scale-invariant
   across nodes / horizons / regimes::

       ε_t = max{q_lo - y, y - q_hi} / (q_hi - q_lo + δ)

2. **Sliding-window correction** (Eq. 22-23).  Instead of one fixed
   ``Q`` computed on the validation split, the correction is the
   ``(1-α̃_t)`` empirical quantile of the *most recent* ``m``
   nonconformity scores, and the interval is widened multiplicatively by
   the raw width::

       Q_t = Quantile_{1-α̃_t}(ℰ_t)
       Ĉ(X_t) = [q_lo - Q_t·w_t ,  q_hi + Q_t·w_t] ,   w_t = q_hi - q_lo

3. **Online feedback** (Eq. 24).  The effective miscoverage α̃ is updated
   after every step from the realised coverage, giving long-run coverage
   convergence to ``1-α`` (Eq. 25)::

       α̃_{t+1} = α̃_t + γ ( α - 𝟙{ Y_t ∉ Ĉ(X_t) } )

The validation split seeds the sliding window so the very first test
steps already have a meaningful correction; from there the window slides
over the test stream in temporal order.
"""

import numpy as np
import torch

from base.CQR_engine import CQR_Engine


class ACQR_Engine(CQR_Engine):
    DEFAULT_METRICS = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "IS", "COV"]

    def __init__(
        self,
        acqr_window: int = 100,
        acqr_gamma: float = 0.005,
        acqr_delta: float = 1e-6,
        **args,
    ):
        """
        Args:
            acqr_window: ``m`` — number of recent nonconformity scores kept
                in the sliding window used to compute ``Q_t``.
            acqr_gamma: ``γ`` — feedback learning rate for the online
                miscoverage update.
            acqr_delta: ``δ`` — numerical-stability floor in the
                width-normalised nonconformity denominator.
        """
        super().__init__(**args)
        # Allow CLI overrides via args (set in main.py's add_args).
        self.window = int(getattr(self.args, "acqr_window", acqr_window))
        self.gamma = float(getattr(self.args, "acqr_gamma", acqr_gamma))
        self.delta = float(getattr(self.args, "acqr_delta", acqr_delta))

        # Seed scores for the sliding window, filled by calibrate().
        self._seed_scores = None

    # -- locally adaptive nonconformity ------------------------------------

    def _nonconformity(self, lower, upper, label):
        """Width-normalised CQR score (Eq. 21), NaN-safe per element."""
        width = (upper - lower).clamp(min=0.0) + self.delta
        return torch.maximum(lower - label, label - upper) / width

    @staticmethod
    def _window_quantile(scores, level):
        """``level`` empirical quantile of a 1-D score buffer (NaN-safe)."""
        scores = scores[~torch.isnan(scores)]
        n = scores.numel()
        if n == 0:
            return 0.0
        level = float(min(max(level, 0.0), 1.0))
        return float(torch.quantile(scores, level))

    # -- calibration: seed the sliding window from validation --------------

    def calibrate(self, mode="val"):
        """Collect width-normalised nonconformity scores on the held-out
        split and store them (in temporal order) to seed the online window.

        Also records a single static correction in ``self._conformal_q`` so
        the checkpoint stays compatible with the parent class and a coarse
        non-adaptive fallback is always available.
        """
        lower, upper, label = self._collect_quantiles(mode)
        scores = self._nonconformity(lower, upper, label)

        # Flatten to a temporally ordered 1-D stream (samples are already in
        # split order; per-sample we ravel node/feature/horizon).
        seed = scores.reshape(scores.shape[0], -1)
        self._seed_scores = seed

        # Static fallback Q (un-normalised correction is meaningless here, so
        # store the width-normalised window quantile at the nominal level).
        flat = seed.reshape(-1)
        q = torch.tensor(self._window_quantile(flat, 1.0 - self.alpha))
        self.register_conformal(q)

        n = int((~torch.isnan(flat)).sum().item())
        self._logger.info(
            f"AS-CQR seed on '{mode}' (alpha={self.alpha}, window={self.window}, "
            f"gamma={self.gamma}, n={n}): Q0 = {float(q):.4f}"
        )
        return self._conformal_q

    # -- online sequential calibration over the test stream ----------------

    def _adaptive_intervals(self, lowers, uppers, labels):
        """Walk the test stream in order, producing calibrated intervals.

        At step ``t``: compute ``Q_t`` from the current window, widen the
        interval by ``Q_t * w_t``, score realised coverage, slide the window
        with the new (width-normalised) nonconformity score, and update the
        effective miscoverage ``α̃`` from the coverage feedback.

        Shapes: ``lowers/uppers/labels`` are ``(B, T, N, F)`` on CPU.
        Returns calibrated ``(lowers, uppers)`` of the same shape.
        """
        B = lowers.shape[0]
        widths = (uppers - lowers).clamp(min=0.0)

        # Initialise the window from the validation seed (most recent m rows).
        if self._seed_scores is not None and self._seed_scores.numel() > 0:
            window = list(self._seed_scores[-self.window:].reshape(-1))
        else:
            window = []

        alpha_t = self.alpha
        out_lower = torch.empty_like(lowers)
        out_upper = torch.empty_like(uppers)
        alpha_trace = []

        for t in range(B):
            if window:
                buf = torch.stack([w if torch.is_tensor(w) else torch.tensor(w)
                                   for w in window])
                Q_t = self._window_quantile(buf, 1.0 - alpha_t)
            else:
                Q_t = float(self._conformal_q) if self._conformal_q is not None else 0.0

            corr = Q_t * widths[t]
            lo_t = lowers[t] - corr
            hi_t = uppers[t] + corr
            out_lower[t] = lo_t
            out_upper[t] = hi_t

            # realised coverage indicator over this step's elements
            y = labels[t]
            valid = ~torch.isnan(y)
            inside = ((y >= lo_t) & (y <= hi_t))[valid]
            miscover = 1.0 - (inside.float().mean().item() if inside.numel() else 1.0)

            # slide window with new nonconformity scores
            new_scores = self._nonconformity(lowers[t], uppers[t], y)
            new_scores = new_scores[~torch.isnan(new_scores)].reshape(-1)
            window.extend(new_scores.tolist())
            if len(window) > self.window:
                window = window[-self.window:]

            # online feedback update of the effective miscoverage (Eq. 24)
            alpha_t = float(np.clip(
                alpha_t + self.gamma * (self.alpha - miscover), 1e-4, 1.0 - 1e-4
            ))
            alpha_trace.append(alpha_t)

        self._logger.info(
            f"AS-CQR online calibration: steps={B}, "
            f"alpha_tilde range=[{min(alpha_trace):.4f}, {max(alpha_trace):.4f}], "
            f"final={alpha_trace[-1]:.4f}"
        )
        return out_lower, out_upper

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        """Same flow as CQR_Engine.evaluate, but the conformal widening is
        the adaptive sequential procedure instead of a static ``±Q``."""
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            if self._seed_scores is None:
                self.calibrate()

        self.model.eval()

        mids, lowers, uppers, labels = [], [], [], []
        with torch.no_grad():
            loader_key = "test_loader" if mode == "export" else f"{mode}_loader"
            for X, label in self._dataloader[loader_key].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                lower, mid, upper, label = self._forward_quantiles(X, label)
                mids.append(mid.cpu())
                lowers.append(lower.cpu())
                uppers.append(upper.cpu())
                labels.append(label.cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        # During training-time validation we only need the pinball/quantile
        # metrics; the adaptive walk is reserved for the test stream.
        if mode == "val":
            self._compute_per_horizon(mids, lowers, uppers, labels, "valid")
            return

        lowers, uppers = self._adaptive_intervals(lowers, uppers, labels)
        self._compute_per_horizon(mids, lowers, uppers, labels, "test")

        if not train_test:
            with self._logger.no_time():
                self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
            for msg in self.metric.get_test_msg():
                self._logger.info(msg)

        if export:
            self.save_result(mids, lowers, uppers, labels)

    def _compute_per_horizon(self, mids, lowers, uppers, labels, mode_name):
        mask_value = torch.tensor(torch.nan)
        for h in range(mids.shape[1]):
            self.metric.compute_one_batch(
                mids[:, h : h + 1],
                labels[:, h : h + 1],
                mask_value,
                mode_name,
                lower=lowers[:, h : h + 1],
                upper=uppers[:, h : h + 1],
                **self._quantile_kw(),
            )

    # -- checkpoint: persist the validation seed + ACQR hyper-params -------

    def save_model(self, save_path):
        import os

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        q = self._conformal_q
        state = {
            "model": self.model.state_dict(),
            "conformal_q": q.cpu() if torch.is_tensor(q) else q,
            "cqr_mode": self.cqr_mode,
            "alpha": self.alpha,
            "acqr_seed_scores": (
                self._seed_scores.cpu() if self._seed_scores is not None else None
            ),
            "acqr_window": self.window,
            "acqr_gamma": self.gamma,
            "acqr_delta": self.delta,
        }
        torch.save(state, os.path.join(save_path, self._time_model))

    def _load_state_dict(self, checkpoint):
        super()._load_state_dict(checkpoint)
        if isinstance(checkpoint, dict) and checkpoint.get("acqr_seed_scores") is not None:
            self._seed_scores = checkpoint["acqr_seed_scores"]
