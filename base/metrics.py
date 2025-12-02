import numpy as np
import properscoring as ps
import torch
from torch.distributions.laplace import Laplace
from torch.distributions.log_normal import LogNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import torch.nn.functional as F


class Metrics:
    def __init__(self, loss_func, metric_lst, horizon=1, early_stop_method="MAE"):
        self.dic = {
            "MAE": masked_mae,
            "MSE": masked_mse,
            "MAPE": masked_mape,
            "RMSE": masked_rmse,
            "MPIW": masked_mpiw,
            "CRPS": masked_crps,
            "WINK": masked_wink,
            "COV": masked_coverage,
            "KL": masked_kl,
            "MGAU": mnormal_loss,
            "Quantile": masked_quantile,
            "IS": masked_IS,
        }
        self.horizon = horizon

        cleaned_metrics = []
        seen = set()
        for m in metric_lst:
            if m not in self.dic:
                raise ValueError(f"Unsupported metric: {m}")
            if m not in seen:
                cleaned_metrics.append(m)
                seen.add(m)

        if loss_func not in seen:
            cleaned_metrics.insert(0, loss_func)
            seen.add(loss_func)

        self.loss_name = loss_func
        self.metric_lst = cleaned_metrics
        self.metric_func = [self.dic[i] for i in self.metric_lst]
        self.early_stop_method_index = self.metric_lst.index(self.loss_name)

        self._basic_metrics = {"MAE", "MSE", "MAPE", "RMSE", "KL", "CRPS"}
        self._scale_metrics = {"MGAU"}
        self._interval_metrics = {"MPIW"}
        self._interval_with_target = {"WINK", "COV", "IS"}
        self._quantile_metric = {"Quantile"}

        self.N = len(self.metric_lst)  # loss function
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]

        self.train_msg = None
        self.test_msg = None
        self.formatter()

    def formatter(self):
        def _section(prefix):
            return [f"{prefix} {m}: {{:.3f}}, " for m in self.metric_lst]

        msg_parts = ["Epoch: {:d}, "]
        msg_parts.extend(_section("Tr"))
        msg_parts.extend(_section("V"))
        msg_parts.extend(_section("Te"))
        msg_parts.append(
            "LR: {:.4e}, Tr Time: {:.3f} s, V Time: {:.3f} s, Te Time: {:.3f} s"
        )
        self.train_msg = "".join(msg_parts)

    def compute_one_batch(self, preds, labels, null_val, mode="train", **kwargs):
        grad_res = None
        res_storage = {
            "train": self.train_res,
            "valid": self.valid_res,
            "test": self.test_res,
        }
        current_storage = res_storage.get(mode, self.test_res)

        null_tensor = self._align_tensor(null_val, preds)

        for i, fname in enumerate(self.metric_lst):
            func = self.metric_func[i]
            if fname in self._basic_metrics:
                res = func(preds, labels, null_tensor)
            elif fname in self._scale_metrics:
                res = func(
                    preds, labels, null_tensor, self._require_kw("scale", kwargs)
                )
            elif fname in self._interval_metrics:
                lower, upper = self._require_interval(kwargs, preds)
                res = func(lower, upper, null_tensor)
            elif fname in self._interval_with_target:
                lower, upper = self._require_interval(kwargs, labels)
                res = func(lower, upper, labels, alpha=kwargs.get("alpha", 0.1))
            elif fname in self._quantile_metric:
                lower, upper = self._require_interval(kwargs, preds)
                res = func(lower, preds, upper, labels)
            else:
                raise ValueError(f"Invalid metric name: {fname}")

            if fname == self.loss_name and mode == "train":
                grad_res = res

            current_storage[i].append(self._to_scalar(res))

        return grad_res

    @staticmethod
    def _align_tensor(value, reference):
        if torch.is_tensor(value):
            return value.to(device=reference.device, dtype=reference.dtype)
        return torch.tensor(value, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _require_kw(key, storage):
        if key not in storage:
            raise KeyError(f"Metric computation requires '{key}' in kwargs.")
        return storage[key]

    def _require_interval(self, storage, reference):
        lower = self._align_tensor(self._require_kw("lower", storage), reference)
        upper = self._align_tensor(self._require_kw("upper", storage), reference)
        return lower, upper

    @staticmethod
    def _to_scalar(value):
        if torch.is_tensor(value):
            return value.detach().item()
        return float(value)

    def get_loss(self, mode="valid", method="MAE"):
        index_ = self.metric_lst.index(method)

        if mode == "train":
            return self.train_res[index_]
        elif mode == "valid":
            return self.valid_res[index_]
        else:
            return self.test_res[index_]

    def get_valid_loss(self):
        return np.mean(self.valid_res[self.early_stop_method_index])

    def get_test_loss(self):
        return np.mean(self.test_res[self.early_stop_method_index])

    def get_epoch_msg(self, epoch, lr, training_time, valid_time, test_time):
        train_lst = [np.mean(i) for i in self.train_res]
        valid_lst = [np.mean(i) for i in self.valid_res]
        test_lst = [np.mean(i) for i in self.test_res]

        msg = self.train_msg.format(
            epoch,
            *train_lst,
            *valid_lst,
            *test_lst,
            lr,
            training_time,
            valid_time,
            test_time,
        )

        self._reset_metrics()
        return msg

    def _reset_metrics(self):
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]

    def get_test_msg(self):
        metric_fmt = ", ".join([f"{m}: {{:.3f}}" for m in self.metric_lst])
        msgs = []

        for i in range(self.horizon):
            test_lst = [k[i] for k in self.test_res]
            msg = f"Test Horizon: {i + 1}, " + metric_fmt.format(*test_lst)
            msgs.append(msg)

        avg_lst = [np.mean(i) for i in self.test_res]
        msgs.append("Average: " + metric_fmt.format(*avg_lst))

        self.test_res = [[] for _ in range(self.N)]
        return msgs

    def export(self):
        return self.test_res


def get_mask(labels, null_val):
    if not torch.is_tensor(null_val):
        null_val = torch.tensor(null_val, device=labels.device, dtype=labels.dtype)
    else:
        null_val = null_val.to(device=labels.device, dtype=labels.dtype)

    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    # mask /= torch.mean(mask)
    # mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def get_mask_mean(loss, labels, null_val):
    mask = get_mask(labels, null_val)
    valid_count = torch.sum(mask)
    if valid_count == 0:
        return torch.tensor(0.0, device=loss.device)
    loss = loss * mask

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.sum(loss) / valid_count


def masked_pinball(preds, labels, null_val, quantile):
    error = preds - labels
    abs_error = torch.abs(error)
    loss = torch.where(error < 0, quantile * abs_error, (1 - quantile) * abs_error)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val):
    assert preds.shape == labels.shape, f"preds: {preds.shape}, labels: {labels.shape}"
    loss = (preds - labels) ** 2
    loss = get_mask_mean(loss, labels, null_val)
    return loss


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    # print("preds:", preds.shape, "labels:", labels.shape)
    assert preds.shape == labels.shape, f"preds: {preds.shape}, labels: {labels.shape}"
    loss = torch.abs(preds - labels)
    loss = get_mask_mean(loss, labels, null_val)
    return loss


def masked_mape(preds, labels, null_val):
    assert preds.shape == labels.shape, f"preds: {preds.shape}, labels: {labels.shape}"
    loss = torch.abs(preds - labels) / torch.abs(labels)
    loss = get_mask_mean(loss, labels, labels.new_tensor(0.0))
    return loss * 100


def masked_kl(preds, labels, null_val):
    assert preds.shape == labels.shape, f"preds: {preds.shape}, labels: {labels.shape}"
    loss = labels * torch.log((labels + 1e-5) / (preds + 1e-5))
    loss = get_mask_mean(loss, labels, null_val)
    return loss


def masked_mpiw(lower, upper, null_val=None):
    return torch.mean(upper - lower)


def masked_wink(lower, upper, labels, alpha=0.1):
    zero = torch.tensor(0.0, device=lower.device)
    score = upper - lower
    score += (2 / alpha) * torch.maximum(lower - labels, zero)
    score += (2 / alpha) * torch.maximum(labels - upper, zero)
    return torch.mean(score)


def masked_coverage(lower, upper, labels, alpha=None):
    in_the_range = torch.sum((labels >= lower) & (labels <= upper))
    coverage = in_the_range / labels.numel() * 100
    return coverage


def masked_IS(lower, upper, labels, alpha=0.1):
    """
    Compute Interval Score (Gneiting & Raftery) for a batch of predictions.
    Handles non-contiguous tensors.
    """
    lower = lower.reshape(-1)
    upper = upper.reshape(-1)
    labels = labels.reshape(-1)

    width = upper - lower

    below = (labels < lower).float()
    above = (labels > upper).float()
    penalty_below = (lower - labels) * below
    penalty_above = (labels - upper) * above

    interval_score = (
        width + (2.0 / alpha) * penalty_below + (2.0 / alpha) * penalty_above
    )
    return interval_score.mean()


def masked_nonconf(lower, upper, labels):
    return torch.maximum(lower - labels, labels - upper)


def masked_mpiw_ens(preds, labels, null_val):
    # mask = get_mask(labels, null_val)

    m = torch.mean(preds, dim=list(range(1, preds.dim())))
    # print(torch.min(preds),torch.quantile(m, 0.05),torch.mean(preds),torch.quantile(m, 0.95),torch.max(preds))

    upper_bound = torch.quantile(m, 0.95)
    lower_bound = torch.quantile(m, 0.05)
    loss = upper_bound - lower_bound

    return torch.mean(
        loss
    )  # -torch.mean(torch.quantile(m, 0.8)-torch.quantile(m, 0.2))


def compute_all_metrics(preds, labels, null_val, lower=None, upper=None):
    mae = masked_mae(preds, labels, null_val)
    mape = masked_mape(preds, labels, null_val)
    rmse = masked_rmse(preds, labels, null_val)

    crps = masked_crps(preds, labels, null_val)
    mpiw = masked_mpiw_ens(preds, labels, null_val)
    kl = masked_kl(preds, labels, null_val)

    res = [mae, rmse, mape, kl, mpiw, crps]

    if lower is not None:
        res[4] = masked_mpiw(lower, upper, null_val)
        wink = masked_wink(lower, upper, labels)
        cov = masked_coverage(lower, upper, labels)
        res = res + [wink, cov]

    return res


def nb_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    n, p, pi = preds
    pi = torch.clip(pi, 1e-3, 1 - 1e-3)
    p = torch.clip(p, 1e-3, 1 - 1e-3)

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    lambda_ = 1e-4

    L_yeq0 = torch.log(pi_yeq0 + lambda_) + torch.log(
        lambda_ + (1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0)
    )
    L_yg0 = (
        torch.log(1 - pi_yg0 + lambda_)
        + torch.lgamma(n_yg0 + yg0)
        - torch.lgamma(yg0 + 1)
        - torch.lgamma(n_yg0 + lambda_)
        + n_yg0 * torch.log(p_yg0 + lambda_)
        + yg0 * torch.log(1 - p_yg0 + lambda_)
    )

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def nb_nll_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    n, p, pi = preds

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    index1 = p_yg0 == 1
    p_yg0[index1] = torch.tensor(0.9999)
    index2 = pi_yg0 == 1
    pi_yg0[index2] = torch.tensor(0.9999)
    index3 = pi_yeq0 == 1
    pi_yeq0[index3] = torch.tensor(0.9999)
    index4 = pi_yeq0 == 0
    pi_yeq0[index4] = torch.tensor(0.001)

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = (
        torch.log(1 - pi_yg0)
        + torch.lgamma(n_yg0 + yg0)
        - torch.lgamma(yg0 + 1)
        - torch.lgamma(n_yg0)
        + n_yg0 * torch.log(p_yg0)
        + yg0 * torch.log(1 - p_yg0)
    )

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def gaussian_nll_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds
    var = torch.pow(scale, 2)
    loss = (labels - loc) ** 2 / var + torch.log(2 * torch.pi * var)

    # pi = torch.acos(torch.zeros(1)).item() * 2
    # loss = 0.5 * (torch.log(2 * torch.pi * var) + (torch.pow(labels - loc, 2) / var))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def laplace_nll_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds
    loss = torch.log(2 * scale) + torch.abs(labels - loc) / scale

    # d = torch.distributions.poisson.Poisson
    # loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def mnormal_loss(preds, labels, null_val, scales):
    mask = get_mask(labels, null_val)

    loc, scale = preds, scales

    dis = MultivariateNormal(loc=loc, covariance_matrix=scale)
    loss = dis.log_prob(labels)

    if loss.shape == mask.shape:
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # loss = -torch.sum(loss)
    loss = -torch.mean(loss)
    return loss


def mnb_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    mu, r = preds

    term1 = torch.lgamma(labels + r) - torch.lgamma(r) - torch.lgamma(labels + 1)
    term2 = r * torch.log(r) + labels * torch.log(mu)
    term3 = -(labels + r) * torch.log(r + mu)
    loss = term1 + term2 + term3

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def normal_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds
    d = Normal(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def lognormal_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds

    dis = LogNormal(loc, scale)
    loss = dis.log_prob(labels + 0.000001)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def tnormal_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds

    d = Normal(loc, scale)
    prob0 = d.cdf(torch.Tensor([0]).to(labels.device))
    loss = d.log_prob(labels) - torch.log(1 - prob0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def laplace_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds

    d = Laplace(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def masked_crps(preds, labels, null_val):
    _ = get_mask(labels, null_val)  # keep API for consistency
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    loss = ps.crps_ensemble(labels_np, preds_np)
    return torch.tensor(loss.mean(), device=preds.device, dtype=preds.dtype)


def masked_quantile(
    y_lower,
    y_middle,
    y_upper,
    y_true,
    q_lower=0.05,
    q_upper=0.95,
    q_middle=0.5,
    lam=1.0,
):
    mask = get_mask(
        y_true,
        torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype),
    )
    valid = mask.sum()
    if valid.item() == 0:
        return y_true.new_tensor(0.0)

    quantiles = y_true.new_tensor([q_lower, q_middle, q_upper]).view(
        -1, *[1] * y_true.ndim
    )
    preds = torch.stack([y_lower, y_middle, y_upper], dim=0)
    errors = y_true.unsqueeze(0) - preds

    positive = errors >= 0
    pinball = torch.where(positive, quantiles * errors, (quantiles - 1) * errors)
    pinball = pinball * mask.unsqueeze(0)
    quantile_loss = pinball.sum() / valid * pinball.shape[0]

    monotonic_penalty = F.relu(y_lower - y_middle) + F.relu(y_middle - y_upper)
    monotonic_penalty = monotonic_penalty * mask
    monotonic_loss = monotonic_penalty.sum() / valid

    return quantile_loss + lam * monotonic_loss


if __name__ == "__main__":
    f = "Epoch: {:d}, T Loss: {:.3f},T Loss: {:.3f},T Loss: {:.3f}"
    print(f.format(1, *[1, 1, 1]))
