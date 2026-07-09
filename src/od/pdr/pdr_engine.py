import torch

from base.engine import BaseEngine_OD
from base.metrics import get_mask


def zinb_mean(n, p, pi):
    """Point prediction (expected value) of the zero-inflated NB, verbatim
    from ``ZINB/od_experiment_utils.py``."""
    p = p.clamp(1e-6, 1.0 - 1e-6)
    pi = pi.clamp(1e-6, 1.0 - 1e-6)
    return (1.0 - pi) * n * (1.0 - p) / p


def zinb_nll(y, n, p, pi, null_val=float("nan")):
    """Zero-Inflated Negative Binomial negative log-likelihood, ported verbatim
    from ``ZINB/od_experiment_utils.py`` (the PDR-ZINB prototype's own loss),
    with NaN-masking added so it plugs into this framework's OD data (which
    may mark invalid cells with NaN, unlike the standalone prototype).

    Uses ``torch.logaddexp`` for the zero-count term
    ``log(pi + (1-pi)*p**n) = logaddexp(log(pi), log1p(-pi) + n*log(p))``
    rather than ``log(pi + (1-pi)*exp(...) + eps)`` — numerically more stable
    for very negative ``n*log(p)``.
    """
    mask = get_mask(y, null_val)
    p = p.clamp(1e-6, 1.0 - 1e-6)
    pi = pi.clamp(1e-6, 1.0 - 1e-6)
    n = n.clamp_min(1e-6)
    y = torch.clamp(y, min=0.0)

    nb_log_prob = (
        torch.lgamma(n + y)
        - torch.lgamma(y + 1.0)
        - torch.lgamma(n)
        + n * torch.log(p)
        + y * torch.log1p(-p)
    )
    zero_log_prob = torch.logaddexp(torch.log(pi), torch.log1p(-pi) + nb_log_prob)
    nonzero_log_prob = torch.log1p(-pi) + nb_log_prob
    log_prob = torch.where(y == 0.0, zero_log_prob, nonzero_log_prob)

    nll = -log_prob
    count = mask.sum()
    if count == 0:
        return torch.tensor(0.0, device=nll.device)
    out = nll * mask
    out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    return out.sum() / count


class PDR_Engine(BaseEngine_OD):
    """Engine for PDR (PDR-ZINB: zero-inflated negative binomial OD demand),
    ported from the standalone ``ZINB/main_pdr_zinb_od.py`` /
    ``test_pdr_zinb_od.py`` prototype.

    The model returns the distribution parameters ``(n, p, pi)`` rather than a
    point tensor.  This engine:

      * trains on the ZINB negative log-likelihood defined in this module
        (ported verbatim from ``ZINB/od_experiment_utils.py``), computed in
        the original count space — only the count labels are
        inverse-transformed, never the ``n``/``p``/``pi`` model outputs;
      * reports point metrics (MAE / RMSE / MAPE) against the ZINB mean
        ``E[y] = (1-pi)·n·(1-p)/p``, likewise ported verbatim;
      * drives early-stopping / best-checkpoint selection off the validation
        NLL itself (``loss_fn="NLL"``), matching the prototype's own
        best-checkpoint-by-val-loss behaviour.

    Everything else (checkpointing, per-horizon test aggregation, export) is
    inherited from :class:`base.engine.BaseEngine_OD`.
    """

    def _to_counts(self, tensor):
        if self._normalize:
            tensor = self._inverse_transform(tensor, device=self._device.type)
        return tensor

    # -- training -------------------------------------------------------

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
            loss = zinb_nll(label_c, n, p, pi, null_val=mask_value)

            with torch.no_grad():
                pred = zinb_mean(n, p, pi)
                self.metric.compute_one_batch(pred, label_c, mask_value, "train", value=loss)

            loss.backward()
            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1

    # -- evaluation -------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        preds, labels, ns, ps, pis = [], [], [], [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                n, p, pi = self.model(X)
                pred = zinb_mean(n, p, pi)
                label = self._to_counts(label)

                if mode == "val":
                    mask_value = self._mask_value.to(pred.device)
                    nll = zinb_nll(label, n, p, pi, null_val=mask_value)
                    self.metric.compute_one_batch(
                        pred, label, mask_value, "valid", value=nll
                    )
                else:
                    preds.append(self._collect(pred).cpu())
                    labels.append(self._collect(label).cpu())
                    ns.append(self._collect(n).cpu())
                    ps.append(self._collect(p).cpu())
                    pis.append(self._collect(pi).cpu())

        if mode == "val":
            return

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        ns = torch.cat(ns, dim=0)
        ps = torch.cat(ps, dim=0)
        pis = torch.cat(pis, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                nll = zinb_nll(
                    self._horizon_slice(labels, i),
                    self._horizon_slice(ns, i),
                    self._horizon_slice(ps, i),
                    self._horizon_slice(pis, i),
                    null_val=mask_value,
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
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(preds, labels)
