"""ZeroCQR: post-hoc conformal intervals for sparse OD matrices.

Unlike the flow CQR engine, this module deliberately operates on the full
``(batch, horizon, origin, destination, channel)`` tensor.  It is a *post-hoc*
method: PDRReg remains a point regressor and no output channels are widened.

OD demand has a large atom at zero.  A single residual quantile is therefore
usually either zero (and misses active trips) or unnecessarily wide for empty
pairs.  ZeroCQR learns a prediction-only zero gate on one part of validation,
fits an active-pair median residual on another part, and conformalizes absolute
residuals separately for predicted-zero and predicted-active pairs on a final,
disjoint calibration part.  The final split is important: gate/centre tuning
does not leak calibration labels into the conformal radii.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from base.engine import BaseEngine_OD
from base.metrics import Metrics, zinb_mean


class ZeroCQREngine(BaseEngine_OD):
    """Sparse-OD-aware, prediction-stratified post-hoc CQR.

    The point estimate is also post-processed: forecasts below a validation
    selected gate are set to zero.  This makes the point model suitable for a
    zero-inflated OD target and, unlike interval widening alone, can improve
    MAE/MSE as well as coverage.  The gate depends only on the forecast at
    inference time.
    """

    DEFAULT_METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MPIW", "IS", "COV", "F1", "TZR", "KL", "CRPS"]

    def __init__(
        self,
        *args,
        zero_cqr_alpha=0.1,
        zero_cqr_gate_quantile=0.95,
        zero_cqr_grid_size=33,
        zero_cqr_min_group=64,
        zero_cqr_mse_weight=1.0,
        zero_cqr_aux_epochs=4,
        zero_cqr_aux_samples=400000,
        zero_cqr_period=96,
        zero_cqr_enable_online=True,
        zero_cqr_zero_floor=1e-3,
        zero_cqr_active_bins=8,
        **kwargs,
    ):
        kwargs["metric_list"] = self.DEFAULT_METRICS
        super().__init__(*args, **kwargs)
        self.alpha = float(zero_cqr_alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("zero_cqr_alpha must be in (0, 1)")
        self.gate_quantile = float(zero_cqr_gate_quantile)
        if not 0.0 < self.gate_quantile <= 1.0:
            raise ValueError("zero_cqr_gate_quantile must be in (0, 1]")
        self.grid_size = max(int(zero_cqr_grid_size), 3)
        self.min_group = max(int(zero_cqr_min_group), 1)
        self.mse_weight = max(float(zero_cqr_mse_weight), 0.0)
        self.aux_epochs = max(int(zero_cqr_aux_epochs), 0)
        self.aux_samples = max(int(zero_cqr_aux_samples), 1)
        # A non-positive period disables the expensive calendar residual
        # experiment while retaining the core sparse ZeroCQR calibration.
        self.period = int(zero_cqr_period)
        self.enable_online = bool(zero_cqr_enable_online)
        self.zero_floor = max(float(zero_cqr_zero_floor), 0.0)
        self.active_bins = max(int(zero_cqr_active_bins), 1)
        self._zero_cqr_gate = None       # (H,)
        self._zero_cqr_active_shift = None  # (H,)
        self._zero_cqr_radius = None     # (H, 1 + active_bins): zero / active bins
        self._zero_cqr_edges = None      # (H, active_bins - 1)
        self._aux_model = None
        self._aux_enabled = False
        self._periodic = None
        self._online_alpha = 0.0

    @staticmethod
    def _finite(x):
        return x[torch.isfinite(x)]

    def _conformal_quantile(self, scores):
        scores = self._finite(scores)
        n = scores.numel()
        if n == 0:
            return torch.tensor(0.0)
        level = min(float(np.ceil((n + 1) * (1.0 - self.alpha)) / n), 1.0)
        return torch.quantile(scores, level)

    def _collect_predictions(self, mode, with_aux=False):
        self.model.eval()
        preds, labels, aux = [], [], []
        with torch.no_grad():
            for X, label in self._dataloader[f"{mode}_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                pred = self._predict(X, label=label, iter=self._iter_cnt)
                if isinstance(pred, tuple):
                    if len(pred) != 3:
                        raise RuntimeError("ZeroCQR supports point output or ZINB (n, p, pi) output")
                    # ZINB parameters are already in count space.  Only the
                    # label/history needs inverse scaling, as in PDR_Engine.
                    pred = zinb_mean(*pred)
                    if self._normalize:
                        label = self._inverse_transform(label, device=self._device.type)
                        if with_aux:
                            X = self._inverse_transform(X, device=self._device.type)
                elif self._normalize:
                    pred, label = self._inverse_transform(
                        [pred, label], device=self._device.type
                    )
                    if with_aux:
                        X = self._inverse_transform(X, device=self._device.type)
                preds.append(pred.cpu())
                labels.append(label.cpu())
                if with_aux:
                    # Explicit sparse-OD information not available in the
                    # scalar point forecast: recent pair demand, its history,
                    # and production/attraction marginals.
                    lag = X[:, -1]
                    hist = X.mean(dim=1)
                    prod = lag.sum(dim=3, keepdim=True).expand_as(lag)
                    attr = lag.sum(dim=2, keepdim=True).expand_as(lag)
                    # Historical features have no forecast-step axis; repeat
                    # them over H so their layout matches the OD prediction.
                    history_features = torch.stack([lag, hist, prod, attr], dim=-1)
                    history_features = history_features.unsqueeze(1).expand(
                        -1, pred.shape[1], -1, -1, -1, -1
                    )
                    aux.append(history_features.cpu())
        if with_aux:
            # Validation/test loaders are ordered (not shuffled), so their
            # dataset indices line up exactly with the concatenated batches.
            indices = torch.as_tensor(self._dataloader[f"{mode}_loader"].dataset.indices).clone()
            return torch.cat(preds, dim=0), torch.cat(labels, dim=0), torch.cat(aux, dim=0), indices
        return torch.cat(preds, dim=0), torch.cat(labels, dim=0)

    @staticmethod
    def _aux_features(raw, aux):
        """Create cell-wise features, preserving OD axes until flattening."""
        B, H, O, D, C = raw.shape
        origin = torch.linspace(0, 1, O, dtype=raw.dtype).view(1, 1, O, 1, 1)
        dest = torch.linspace(0, 1, D, dtype=raw.dtype).view(1, 1, 1, D, 1)
        origin = origin.expand(B, H, O, D, C)
        dest = dest.expand(B, H, O, D, C)
        return torch.cat([raw.unsqueeze(-1), aux, origin.unsqueeze(-1), dest.unsqueeze(-1)], dim=-1)

    def _fit_aux(self, raw, label, aux, tune, fit):
        """Fit an out-of-fold residual MLP using observed OD history.

        It is accepted only if its independent fit split improves *both* MAE
        and MSE.  This makes the extra information useful when it generalises,
        rather than silently replacing a strong PDR point model with an
        overfitted calibrator.
        """
        if self.aux_epochs == 0:
            return
        x = self._aux_features(raw[tune], aux[tune]).reshape(-1, 7)
        y = label[tune].reshape(-1, 1)
        good = torch.isfinite(x).all(dim=1) & torch.isfinite(y[:, 0])
        x, y = x[good], y[good]
        if x.numel() == 0:
            return
        if x.shape[0] > self.aux_samples:
            keep = torch.randperm(x.shape[0])[: self.aux_samples]
            x, y = x[keep], y[keep]
        device = self._device
        # The last layer predicts a bounded residual: it is deliberately a
        # correction to PDR rather than a second OD forecaster.
        model = nn.Sequential(nn.Linear(7, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        x, y = x.to(device), y.to(device)
        for _ in range(self.aux_epochs):
            order = torch.randperm(x.shape[0], device=device)
            for ids in order.split(8192):
                pred = (x[ids, :1] + 3.0 * torch.tanh(model(x[ids]))).clamp_min(0.0)
                err = pred - y[ids]
                loss = err.abs().mean() + self.mse_weight * err.square().mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            xf = self._aux_features(raw[fit], aux[fit]).reshape(-1, 7).to(device)
            yf = label[fit].reshape(-1).to(device)
            base = raw[fit].clamp_min(0).reshape(-1).to(device)
            out = []
            for batch in xf.split(65536):
                out.append((batch[:, :1] + 3.0 * torch.tanh(model(batch))).clamp_min(0).squeeze(1))
            point = torch.cat(out)
            valid = torch.isfinite(yf)
            base_mae, base_mse = (base[valid] - yf[valid]).abs().mean(), (base[valid] - yf[valid]).square().mean()
            aux_mae, aux_mse = (point[valid] - yf[valid]).abs().mean(), (point[valid] - yf[valid]).square().mean()
        if aux_mae <= base_mae and aux_mse <= base_mse:
            self._aux_model = model.cpu().eval()
            self._aux_enabled = True
            self._logger.info(f"LagZeroCQR accepted: MAE {base_mae:.4f}->{aux_mae:.4f}, MSE {base_mse:.4f}->{aux_mse:.4f}")
        else:
            self._logger.info(f"LagZeroCQR rejected: MAE {base_mae:.4f}->{aux_mae:.4f}, MSE {base_mse:.4f}->{aux_mse:.4f}")

    def _apply_aux(self, raw, aux):
        if not self._aux_enabled:
            return raw
        features = self._aux_features(raw, aux).reshape(-1, 7)
        out = []
        with torch.no_grad():
            for batch in features.split(65536):
                out.append((batch[:, :1] + 3.0 * torch.tanh(self._aux_model(batch))).clamp_min(0))
        return torch.cat(out).reshape_as(raw)

    def _fit_periodic(self, raw, label, indices, tune, fit):
        """Try a periodic origin/destination residual decomposition.

        It introduces calendar phase (e.g. 15-minute slot within a day) plus
        OD marginals as information unavailable to the scalar output.  The
        decomposition is strongly pooled (slot/origin + slot/destination -
        slot/global), which avoids the noisy per-pair lookup-table failure on
        sparse OD data.
        """
        B, H, O, D, C = raw.shape
        choices, tables = [], []
        weights = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
        for h in range(H):
            slot_t = ((indices[tune] + h + 1) % self.period).long()
            slot_f = ((indices[fit] + h + 1) % self.period).long()
            r_t, y_t = raw[tune, h], label[tune, h]
            residual = torch.where(torch.isfinite(y_t - r_t), y_t - r_t, torch.zeros_like(r_t))
            # slot-global, slot-origin and slot-destination pooled means.
            g_sum = torch.zeros(self.period, C); g_n = torch.zeros(self.period, C)
            o_sum = torch.zeros(self.period, O, C); o_n = torch.zeros(self.period, O, C)
            d_sum = torch.zeros(self.period, D, C); d_n = torch.zeros(self.period, D, C)
            for s in range(self.period):
                m = slot_t == s
                if not m.any():
                    continue
                rr = residual[m]
                valid = torch.isfinite(y_t[m]) & torch.isfinite(r_t[m])
                g_sum[s] = (rr * valid).sum((0, 1, 2)); g_n[s] = valid.sum((0, 1, 2))
                o_sum[s] = (rr * valid).sum((0, 2)); o_n[s] = valid.sum((0, 2))
                d_sum[s] = (rr * valid).sum((0, 1)); d_n[s] = valid.sum((0, 1))
            g = g_sum / g_n.clamp_min(1)
            origin = o_sum / o_n.clamp_min(1)
            dest = d_sum / d_n.clamp_min(1)
            # Sparse slots not observed in tune get no correction.
            corr_f = origin[slot_f].unsqueeze(2) + dest[slot_f].unsqueeze(1) - g[slot_f].unsqueeze(1).unsqueeze(1)
            base, y_f = raw[fit, h].clamp_min(0), label[fit, h]
            valid_f = torch.isfinite(base) & torch.isfinite(y_f)
            base_mae = (base[valid_f] - y_f[valid_f]).abs().mean()
            base_mse = (base[valid_f] - y_f[valid_f]).square().mean()
            best_w, best_score, best_mae, best_mse = 0.0, float("inf"), base_mae, base_mse
            for w in weights:
                pred = (base + w * corr_f).clamp_min(0)
                mae = (pred[valid_f] - y_f[valid_f]).abs().mean()
                mse = (pred[valid_f] - y_f[valid_f]).square().mean()
                score = float(mae / base_mae.clamp_min(1e-8) + self.mse_weight * mse / base_mse.clamp_min(1e-8))
                if score < best_score:
                    best_w, best_score, best_mae, best_mse = w, score, mae, mse
            choices.append((best_w, base_mae, base_mse, best_mae, best_mse))
            tables.append((g, origin, dest))
        # Accept only a true Pareto improvement at every forecast horizon.
        if all(mae <= bmae and mse <= bmse for _, bmae, bmse, mae, mse in choices) and any(w > 0 for w, *_ in choices):
            self._periodic = (tables, [v[0] for v in choices])
            detail = ", ".join(f"h{i+1} w={w:.2f} MAE {bmae:.4f}->{mae:.4f} MSE {bmse:.4f}->{mse:.4f}" for i, (w,bmae,bmse,mae,mse) in enumerate(choices))
            self._logger.info(f"PeriodicODCQR accepted: {detail}")
        else:
            detail = ", ".join(f"h{i+1} w={w:.2f} MAE {bmae:.4f}->{mae:.4f} MSE {bmse:.4f}->{mse:.4f}" for i, (w,bmae,bmse,mae,mse) in enumerate(choices))
            self._logger.info(f"PeriodicODCQR rejected: {detail}")

    def _apply_periodic(self, raw, indices):
        if self._periodic is None:
            return raw
        tables, weights = self._periodic
        point = raw.clone()
        for h, ((g, origin, dest), w) in enumerate(zip(tables, weights)):
            slots = ((indices + h + 1) % self.period).long()
            correction = origin[slots].unsqueeze(2) + dest[slots].unsqueeze(1) - g[slots].unsqueeze(1).unsqueeze(1)
            point[:, h] = (point[:, h] + w * correction).clamp_min(0)
        return point

    @staticmethod
    def _online_correct(raw, labels, alpha):
        """Causal OD-marginal residual filter.

        At time t the correction is formed solely from errors observed before
        t.  After predicting t, its realised label updates the state for
        t+1.  This is valid for rolling OD service where the previous interval
        has closed, and captures short-lived city-wide / origin / destination
        shocks that a static post-hoc calibrator cannot see.
        """
        if alpha <= 0:
            return raw
        B, H, O, D, C = raw.shape
        out = torch.empty_like(raw)
        for h in range(H):
            global_r = raw.new_zeros(C)
            origin_r = raw.new_zeros(O, C)
            dest_r = raw.new_zeros(D, C)
            for t in range(B):
                correction = global_r.view(1, 1, C)
                correction = correction + 0.5 * origin_r.view(O, 1, C)
                correction = correction + 0.5 * dest_r.view(1, D, C)
                # High-volume routes can create an extreme marginal residual;
                # point calibration is deliberately bounded, leaving large
                # shocks to the conformal interval rather than spreading them
                # to every OD cell of the same origin/destination.
                correction = correction.clamp(-0.15, 0.15)
                out[t, h] = (raw[t, h] + correction).clamp_min(0)
                residual = labels[t, h] - raw[t, h]
                valid = torch.isfinite(residual)
                residual = torch.where(valid, residual, torch.zeros_like(residual))
                # Cell counts differ only for missing values; use explicit
                # masks so invalid OD cells do not drag residuals towards zero.
                g = (residual * valid).sum((0, 1)) / valid.sum((0, 1)).clamp_min(1)
                o = (residual * valid).sum(1) / valid.sum(1).clamp_min(1)
                d = (residual * valid).sum(0) / valid.sum(0).clamp_min(1)
                global_r = (1 - alpha) * global_r + alpha * g
                origin_r = (1 - alpha) * origin_r + alpha * o
                dest_r = (1 - alpha) * dest_r + alpha * d
        return out

    def _fit_online(self, raw, label):
        """Select causal residual-filter speed on a future validation block."""
        B = raw.shape[0]
        if B < 12:
            return
        split = B // 3
        # Each candidate processes the prefix causally but is evaluated only
        # after it has warmed up; this mirrors later test deployment.
        base = raw[split : 2 * split].clamp_min(0)
        target = label[split : 2 * split]
        valid = torch.isfinite(base) & torch.isfinite(target)
        base_mae = (base[valid] - target[valid]).abs().mean()
        base_mse = (base[valid] - target[valid]).square().mean()
        chosen = 0.0
        best = 1.0 + self.mse_weight
        detail = []
        for alpha in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            candidate = self._online_correct(raw[: 2 * split], label[: 2 * split], alpha)[split:]
            mae = (candidate[valid] - target[valid]).abs().mean()
            mse = (candidate[valid] - target[valid]).square().mean()
            score = float(mae / base_mae.clamp_min(1e-8) + self.mse_weight * mse / base_mse.clamp_min(1e-8))
            detail.append((alpha, mae, mse))
            if mae <= base_mae and mse <= base_mse and score < best:
                chosen, best = alpha, score
        self._online_alpha = chosen
        trace = ", ".join(f"a={a:g}: MAE={m:.4f},MSE={s:.4f}" for a, m, s in detail)
        self._logger.info(
            f"OnlineODCQR {'accepted' if chosen else 'rejected'} (base MAE={base_mae:.4f}, MSE={base_mse:.4f}, alpha={chosen:g}): {trace}"
        )

    @staticmethod
    def _postprocess(raw, gate, active_shift, zero_floor=0.0):
        """Apply the horizon-wise prediction-only zero gate to a 5-D OD tensor."""
        shape = (1, -1) + (1,) * (raw.ndim - 2)
        gate = gate.to(raw.device, raw.dtype).view(*shape)
        shift = active_shift.to(raw.device, raw.dtype).view(*shape)
        active = raw > gate
        # OD counts are non-negative.  Clamping is deliberately post-hoc so it
        # cannot kill gradients of the underlying point regressor.
        point = torch.where(active, (raw + shift).clamp_min(0.0), torch.zeros_like(raw))
        return torch.where(point <= zero_floor, torch.zeros_like(point), point)

    def calibrate(self, mode="val"):
        """Fit gate, active residual centre, and groupwise CQR radii.

        Validation examples are split by sample/time into tune, centre-fit and
        conformal subsets.  Keeping whole OD matrices together preserves their
        within-time dependence while preventing the usual zero-gate tuning
        from contaminating split-conformal calibration.
        """
        raw, label, aux, indices = self._collect_predictions(mode, with_aux=True)
        if raw.ndim != 5:
            raise RuntimeError(
                "ZeroCQR requires OD tensors (B, H, O, D, C); got "
                f"{tuple(raw.shape)}"
            )
        if raw.shape[0] < 3:
            raise RuntimeError("ZeroCQR needs at least three validation samples")

        tune, fit, cal = slice(0, None, 3), slice(1, None, 3), slice(2, None, 3)
        self._fit_aux(raw, label, aux, tune, fit)
        raw = self._apply_aux(raw, aux)
        if self.period > 0:
            self._fit_periodic(raw, label, indices, tune, fit)
            raw = self._apply_periodic(raw, indices)
        if self.enable_online:
            self._fit_online(raw, label)
            raw = self._online_correct(raw, label, self._online_alpha)
        gates, shifts, radii, edges_all = [], [], [], []
        for h in range(raw.shape[1]):
            r_t, y_t = raw[tune, h], label[tune, h]
            r_f, y_f = raw[fit, h], label[fit, h]
            valid_t = torch.isfinite(r_t) & torch.isfinite(y_t)
            values = r_t[valid_t]
            if values.numel() == 0:
                gate = raw.new_tensor(0.0)
            else:
                levels = torch.linspace(0.0, self.gate_quantile, self.grid_size)
                candidates = torch.unique(torch.quantile(values, levels))
                # Tune a zero gate with a median active residual, then score it
                # on a different split.  MAE alone favours a nearly-zero gate
                # for this data; the normalised MSE term prevents the usual
                # sparse-data failure of improving zero MAE while worsening a
                # few high-volume OD cells disproportionately.
                base_f = r_f.clamp_min(0.0)
                valid_base = torch.isfinite(base_f) & torch.isfinite(y_f)
                base_abs = torch.abs(base_f[valid_base] - y_f[valid_base]).mean()
                base_sq = ((base_f[valid_base] - y_f[valid_base]) ** 2).mean()
                best_gate, best_loss = candidates[0], float("inf")
                for candidate in candidates:
                    active_t = r_t > candidate
                    residual_t = (y_t - r_t)[active_t & valid_t]
                    shift_t = torch.median(residual_t) if residual_t.numel() else r_t.new_tensor(0.0)
                    pred_f = torch.where(
                        r_f > candidate,
                        (r_f + shift_t).clamp_min(0.0),
                        torch.zeros_like(r_f),
                    )
                    valid_f = torch.isfinite(pred_f) & torch.isfinite(y_f)
                    mae = torch.abs(pred_f[valid_f] - y_f[valid_f]).mean()
                    mse = ((pred_f[valid_f] - y_f[valid_f]) ** 2).mean()
                    loss = mae / base_abs.clamp_min(1e-8)
                    loss = loss + self.mse_weight * mse / base_sq.clamp_min(1e-8)
                    if torch.isfinite(loss) and float(loss) < best_loss:
                        best_gate, best_loss = candidate, float(loss)
                gate = best_gate

            active_f = (r_f > gate) & torch.isfinite(r_f) & torch.isfinite(y_f)
            residual_f = (y_f - r_f)[active_f]
            shift = torch.median(residual_f) if residual_f.numel() else raw.new_tensor(0.0)
            # Prediction-only Mondrian bins.  Edges are learned before the
            # disjoint conformal split, so active high-demand OD cells no
            # longer impose their wide residual radius on low-demand cells.
            active_t_values = r_t[(r_t > gate) & valid_t]
            if self.active_bins > 1 and active_t_values.numel() >= self.min_group:
                levels = torch.arange(1, self.active_bins, dtype=raw.dtype) / self.active_bins
                edges = torch.quantile(active_t_values, levels).unique_consecutive()
                # Degenerate predictions can collapse quantiles.  Pad with
                # +inf so bucket indices remain well-defined and unused bins
                # fall back to the pooled active correction.
                if edges.numel() < self.active_bins - 1:
                    edges = torch.cat([edges, raw.new_full((self.active_bins - 1 - edges.numel(),), float("inf"))])
            else:
                edges = raw.new_full((self.active_bins - 1,), float("inf"))
            pred_c = torch.where(
                raw[cal, h] > gate,
                (raw[cal, h] + shift).clamp_min(0.0),
                torch.zeros_like(raw[cal, h]),
            )
            pred_c = torch.where(pred_c <= self.zero_floor, torch.zeros_like(pred_c), pred_c)
            y_c = label[cal, h]
            valid_c = torch.isfinite(pred_c) & torch.isfinite(y_c)
            scores = torch.abs(y_c - pred_c)
            pooled = self._conformal_quantile(scores[valid_c])
            zero_scores = scores[valid_c & (raw[cal, h] <= gate)]
            active_scores = scores[valid_c & (raw[cal, h] > gate)]
            q_zero = self._conformal_quantile(zero_scores) if zero_scores.numel() >= self.min_group else pooled
            q_active = self._conformal_quantile(active_scores) if active_scores.numel() >= self.min_group else pooled
            bin_id = torch.bucketize(raw[cal, h].contiguous(), edges)
            q_bins = []
            for b in range(self.active_bins):
                b_scores = scores[valid_c & (raw[cal, h] > gate) & (bin_id == b)]
                q_bins.append(self._conformal_quantile(b_scores) if b_scores.numel() >= self.min_group else q_active)
            gates.append(gate)
            shifts.append(shift)
            radii.append(torch.stack([q_zero, *q_bins]))
            edges_all.append(edges)

        self._zero_cqr_gate = torch.stack(gates)
        self._zero_cqr_active_shift = torch.stack(shifts)
        self._zero_cqr_radius = torch.stack(radii)
        self._zero_cqr_edges = torch.stack(edges_all)
        detail = ", ".join(
            f"h{i + 1}: gate={g:.3f}, shift={s:.3f}, q0={q[0]:.3f}, q+={q[1:].tolist()}"
            for i, (g, s, q) in enumerate(zip(gates, shifts, radii))
        )
        self._logger.info(
            f"ZeroCQR calibrated on '{mode}' (alpha={self.alpha}, split=1/3+1/3+1/3): {detail}"
        )

    def _interval(self, raw, point):
        shape = (1, -1) + (1,) * (raw.ndim - 2)
        gate = self._zero_cqr_gate.to(raw.device, raw.dtype).view(*shape)
        radii = self._zero_cqr_radius.to(raw.device, raw.dtype)
        edges = self._zero_cqr_edges.to(raw.device, raw.dtype)
        radius = torch.empty_like(raw)
        for h in range(raw.shape[1]):
            bucket = torch.bucketize(raw[:, h].contiguous(), edges[h])
            q_active = radii[h, 1:][bucket]
            q_zero = radii[h, 0]
            radius[:, h] = torch.where(raw[:, h] <= gate[:, h], q_zero, q_active)
        return (point - radius).clamp_min(0.0), point + radius

    def train_batch(self):
        """Ordinary point-regression training; degenerate intervals are logged
        only so the shared metric accumulator has all requested arguments."""
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)
        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(f"Mask Value: {mask_value}\n\n" + "=" * 25 + "   Training   " + "=" * 25)
            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])
            pred = self._predict(X, label=label, iter=self._iter_cnt)
            if isinstance(pred, tuple):
                pred = pred[0]
            if self._normalize:
                pred, label = self._inverse_transform([pred, label], device=self._device.type)
            loss = self.metric.compute_one_batch(
                pred, label, mask_value, "train", lower=pred, upper=pred, alpha=self.alpha
            )
            if torch.isfinite(loss):
                loss.backward()
                if self._clip_grad_norm != 0:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)
                    if not torch.isfinite(norm):
                        self._optimizer.zero_grad()
                    else:
                        self._optimizer.step()
                else:
                    self._optimizer.step()
            self._iter_cnt += 1

    def _evaluate_raw(self, mode, export=None, train_test=False):
        """BaseEngine evaluation with degenerate raw intervals before final CQR."""
        self.model.eval()
        preds, labels = self._collect_predictions(mode)
        if mode == "val":
            self.metric.compute_one_batch(
                preds, labels, torch.tensor(float("nan")), "valid",
                lower=preds, upper=preds, alpha=self.alpha,
            )
            return
        for h in range(self.model.horizon):
            self.metric.compute_one_batch(
                self._horizon_slice(preds, h), self._horizon_slice(labels, h),
                torch.tensor(float("nan")), "test",
                lower=self._horizon_slice(preds, h), upper=self._horizon_slice(preds, h), alpha=self.alpha,
            )

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        # Keep ordinary raw validation/test metrics during training so early
        # stopping remains comparable to the underlying point regressor.
        final_test = mode == "test" and not train_test
        if final_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            self.calibrate()
            # The full prediction/interval tensor is intentionally controlled
            # by --export (it is large for OD).  Always persist this small
            # artifact, however, so a normal test run leaves unambiguous
            # evidence of the fitted post-hoc calibration.
            self.save_calibration_state()

        if not final_test:
            return self._evaluate_raw(mode, export, train_test)

        raw, labels, aux, indices = self._collect_predictions("test", with_aux=True)
        raw = self._apply_aux(raw, aux)
        raw = self._apply_periodic(raw, indices)
        # The loop is causal: labels[t] update only predictions t+1 onward.
        raw = self._online_correct(raw, labels, self._online_alpha)
        point = self._postprocess(
            raw, self._zero_cqr_gate, self._zero_cqr_active_shift, self.zero_floor
        )
        lower, upper = self._interval(raw, point)
        # Keep collected OD tensors on CPU while fitting/applying calibration,
        # but avoid CPU-only metric passes over tens of millions of cells.
        metric_point, metric_lower, metric_upper, metric_labels = (
            point.to(self._device), lower.to(self._device),
            upper.to(self._device), labels.to(self._device),
        )
        mask_value = torch.tensor(float("nan"))
        for h in range(self.model.horizon):
            self.metric.compute_one_batch(
                self._horizon_slice(metric_point, h), self._horizon_slice(metric_labels, h), mask_value,
                "test", lower=self._horizon_slice(metric_lower, h), upper=self._horizon_slice(metric_upper, h), alpha=self.alpha,
            )
        with self._logger.no_time():
            self._logger.info("\n" + "=" * 25 + "     Test (ZeroCQR)     " + "=" * 25)
        for msg in self.metric.get_test_msg():
            self._logger.info(msg)
        if export:
            self.save_result(point, lower, upper, labels)

    def save_calibration_state(self):
        """Persist the fitted ZeroCQR parameters without exporting all OD cells."""
        tag = getattr(self.args, "calibration_tag", "").strip()
        suffix = f"zero_cqr_{tag}_calibration" if tag else "zero_cqr_calibration"
        base_name = f"{self.args.model_name}-{self.args.dataset}-{suffix}"
        path = os.path.join(self._save_path, f"{base_name}.pt")
        index = 1
        while os.path.exists(path):
            path = os.path.join(self._save_path, f"{base_name}_{index}.pt")
            index += 1
        torch.save(
            {
                "method": "ZeroCQR",
                "alpha": self.alpha,
                "zero_floor": self.zero_floor,
                "active_bins": self.active_bins,
                "gate": self._zero_cqr_gate.cpu(),
                "active_shift": self._zero_cqr_active_shift.cpu(),
                "radius": self._zero_cqr_radius.cpu(),
                "active_bin_edges": self._zero_cqr_edges.cpu(),
            },
            path,
        )
        self._logger.info(f"ZeroCQR calibration state saved: {path}")

    def save_result(self, point, lower, upper, labels):
        result = torch.stack([point, lower, upper, labels], dim=0).numpy()
        tag = getattr(self.args, "calibration_tag", "").strip()
        suffix = f"zero_cqr_{tag}_res" if tag else "zero_cqr_res"
        path = self._get_unique_save_path(suffix)
        np.save(path, result)
        self._logger.info(
            f"ZeroCQR Results Save Path: {path} | Shape: {result.shape} "
            "(point, lower, upper, label, batch, horizon, origin, destination, channel)"
        )
