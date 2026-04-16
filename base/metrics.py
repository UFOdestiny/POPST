import numpy as np
import properscoring as ps
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def get_mask(labels, null_val):
    """Boolean-float mask: 1.0 where label is valid, 0.0 where null."""
    if not torch.is_tensor(null_val):
        null_val = torch.tensor(null_val, device=labels.device, dtype=labels.dtype)
    else:
        null_val = null_val.to(device=labels.device, dtype=labels.dtype)
    if torch.isnan(null_val):
        return (~torch.isnan(labels)).float()
    return (labels != null_val).float()


def _masked_mean(loss, mask):
    """Mean of *loss* over positions where *mask* == 1, NaN-safe."""
    count = mask.sum()
    if count == 0:
        return torch.tensor(0.0, device=loss.device)
    out = loss * mask
    out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    return out.sum() / count



# ---------------------------------------------------------------------------
# Basic metrics  (signature: preds, labels, null_val)
# ---------------------------------------------------------------------------

def masked_mae(preds, labels, null_val):
    assert preds.shape == labels.shape, f"preds {preds.shape} vs labels {labels.shape}"
    return _masked_mean(torch.abs(preds - labels), get_mask(labels, null_val))


def masked_mse(preds, labels, null_val):
    assert preds.shape == labels.shape, f"preds {preds.shape} vs labels {labels.shape}"
    return _masked_mean((preds - labels) ** 2, get_mask(labels, null_val))


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape(preds, labels, null_val):
    """MAPE with standard zero-masking (zeros excluded from denominator)."""
    assert preds.shape == labels.shape, f"preds {preds.shape} vs labels {labels.shape}"
    mask = get_mask(labels, labels.new_tensor(0.0))
    loss = torch.abs(preds - labels) / torch.abs(labels).clamp(min=1e-8)
    return _masked_mean(loss, mask) * 100


def masked_kl(preds, labels, null_val):
    assert preds.shape == labels.shape, f"preds {preds.shape} vs labels {labels.shape}"
    loss = labels * torch.log((labels + 1e-5) / (preds + 1e-5))
    return _masked_mean(loss, get_mask(labels, null_val))


def masked_crps(preds, labels, null_val):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    score = ps.crps_ensemble(labels_np, preds_np)
    return torch.tensor(score.mean(), device=preds.device, dtype=preds.dtype)


# ---------------------------------------------------------------------------
# Interval / coverage metrics
# ---------------------------------------------------------------------------

def masked_mpiw(lower, upper, null_val=None):
    return torch.mean(upper - lower)


def masked_wink(lower, upper, labels, alpha=0.1):
    zero = torch.tensor(0.0, device=lower.device)
    score = (upper - lower
             + (2 / alpha) * torch.maximum(lower - labels, zero)
             + (2 / alpha) * torch.maximum(labels - upper, zero))
    return torch.mean(score)


def masked_coverage(lower, upper, labels, alpha=None):
    in_range = ((labels >= lower) & (labels <= upper)).sum()
    return in_range / labels.numel() * 100


def masked_IS(lower, upper, labels, alpha=0.1):
    """Interval Score (Gneiting & Raftery)."""
    lower, upper, labels = lower.reshape(-1), upper.reshape(-1), labels.reshape(-1)
    width = upper - lower
    penalty = ((2.0 / alpha) * (lower - labels) * (labels < lower).float()
               + (2.0 / alpha) * (labels - upper) * (labels > upper).float())
    return (width + penalty).mean()


# ---------------------------------------------------------------------------
# Distribution / ensemble metrics
# ---------------------------------------------------------------------------

def mnormal_loss(preds, labels, null_val, scales):
    """Multivariate normal negative log-likelihood."""
    mask = get_mask(labels, null_val)
    dis = MultivariateNormal(loc=preds, covariance_matrix=scales)
    loss = dis.log_prob(labels)
    if loss.shape == mask.shape:
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return -torch.mean(loss)



def masked_quantile(y_lower, y_middle, y_upper, y_true,
                    q_lower=0.05, q_upper=0.95, q_middle=0.5, lam=1.0):
    mask = get_mask(y_true, y_true.new_tensor(float("nan")))
    valid = mask.sum()
    if valid.item() == 0:
        return y_true.new_tensor(0.0)

    quantiles = y_true.new_tensor([q_lower, q_middle, q_upper]).view(
        -1, *[1] * y_true.ndim
    )
    preds = torch.stack([y_lower, y_middle, y_upper], dim=0)
    errors = y_true.unsqueeze(0) - preds

    pinball = torch.where(errors >= 0, quantiles * errors, (quantiles - 1) * errors)
    pinball = pinball * mask.unsqueeze(0)
    q_loss = pinball.sum() / valid * pinball.shape[0]

    mono = F.relu(y_lower - y_middle) + F.relu(y_middle - y_upper)
    mono_loss = (mono * mask).sum() / valid

    return q_loss + lam * mono_loss



# ---------------------------------------------------------------------------
# Metric registry  (function, call-kind)
# ---------------------------------------------------------------------------

_REGISTRY = {
    "MAE":      (masked_mae,      "basic"),
    "MSE":      (masked_mse,      "basic"),
    "MAPE":     (masked_mape,     "basic"),
    "RMSE":     (masked_rmse,     "basic"),
    "KL":       (masked_kl,       "basic"),
    "CRPS":     (masked_crps,     "basic"),
    "MGAU":     (mnormal_loss,    "scale"),
    "MPIW":     (masked_mpiw,     "interval"),
    "WINK":     (masked_wink,     "interval_target"),
    "COV":      (masked_coverage, "interval_target"),
    "IS":       (masked_IS,       "interval_target"),
    "Quantile": (masked_quantile, "quantile"),
}


def _dispatch(fn, kind, preds, labels, null, kw):
    """Call a metric function according to its kind."""
    if kind == "basic":
        return fn(preds, labels, null)
    if kind == "scale":
        return fn(preds, labels, null, kw["scale"])
    if kind == "interval":
        return fn(kw["lower"], kw["upper"], null)
    if kind == "interval_target":
        return fn(kw["lower"], kw["upper"], labels, alpha=kw.get("alpha", 0.1))
    if kind == "quantile":
        return fn(kw["lower"], preds, kw["upper"], labels)
    raise ValueError(f"Unknown metric kind: {kind}")


def _to_scalar(value):
    return value.detach().item() if torch.is_tensor(value) else float(value)


def _align(value, ref):
    """Ensure *value* is a tensor on the same device/dtype as *ref*."""
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

class Metrics:
    """Accumulates per-batch metrics for train / valid / test splits
    and formats human-readable epoch & test summaries."""

    def __init__(self, loss_func, metric_lst, horizon=1, early_stop_method="MAE"):
        seen, names = set(), []
        for m in metric_lst:
            if m not in _REGISTRY:
                raise ValueError(f"Unknown metric: {m}")
            if m not in seen:
                names.append(m)
                seen.add(m)
        if loss_func not in seen:
            names.insert(0, loss_func)

        self.metric_lst = names
        self.loss_name = loss_func
        self.horizon = horizon
        self.N = len(names)
        self.early_stop_method_index = names.index(loss_func)

        self._funcs = [_REGISTRY[n][0] for n in names]
        self._kinds = [_REGISTRY[n][1] for n in names]
        self._reset_metrics()

    # -- accumulation -------------------------------------------------------

    def _reset_metrics(self):
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res  = [[] for _ in range(self.N)]

    _SPLITS = {"train": "train_res", "valid": "valid_res", "test": "test_res"}

    def compute_one_batch(self, preds, labels, null_val, mode="train", **kw):
        """Compute all metrics for one batch; returns the loss tensor for backprop
        when *mode* == ``'train'``."""
        null = _align(null_val, preds)
        buf = getattr(self, self._SPLITS.get(mode, "test_res"))
        grad_res = None

        for i, (fn, kind) in enumerate(zip(self._funcs, self._kinds)):
            val = _dispatch(fn, kind, preds, labels, null, kw)
            if self.metric_lst[i] == self.loss_name and mode == "train":
                grad_res = val
            buf[i].append(_to_scalar(val))

        return grad_res

    # -- query --------------------------------------------------------------

    def get_loss(self, mode="valid", method="MAE"):
        idx = self.metric_lst.index(method)
        return getattr(self, self._SPLITS.get(mode, "test_res"))[idx]

    def get_valid_loss(self):
        return np.mean(self.valid_res[self.early_stop_method_index])

    def get_test_loss(self):
        return np.mean(self.test_res[self.early_stop_method_index])

    # -- formatting ---------------------------------------------------------

    @staticmethod
    def _fmt_metrics(prefix, names, values):
        return [f"{prefix} {n}: {v:.3f}" for n, v in zip(names, values)]

    def get_epoch_msg(self, epoch, lr, training_time, valid_time, test_time):
        """One-line epoch summary; resets accumulators."""
        parts = [f"Epoch: {epoch}"]
        for pfx, buf in [("Tr", self.train_res), ("V", self.valid_res), ("Te", self.test_res)]:
            parts += self._fmt_metrics(pfx, self.metric_lst, [np.mean(v) for v in buf])
        parts += [f"LR: {lr:.4e}",
                  f"Tr Time: {training_time:.3f} s",
                  f"V Time: {valid_time:.3f} s",
                  f"Te Time: {test_time:.3f} s"]
        self._reset_metrics()
        return ", ".join(parts)

    def get_test_msg(self):
        """Per-horizon test results; resets test accumulator."""
        def _metric_str(values):
            return ", ".join(f"{n}: {v:.3f}" for n, v in zip(self.metric_lst, values))

        msgs = []
        for h in range(self.horizon):
            vals = [buf[h] for buf in self.test_res]
            msgs.append(f"Test Horizon: {h + 1}, {_metric_str(vals)}")

        avgs = [np.mean(buf) for buf in self.test_res]
        msgs.append(f"Average: {_metric_str(avgs)}")

        self.test_res = [[] for _ in range(self.N)]
        return msgs

    def export(self):
        return self.test_res
