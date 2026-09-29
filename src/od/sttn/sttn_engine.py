import torch

from base.engine import BaseEngine_OD
from base.metrics import mnormal_loss


class STTN_Engine(BaseEngine_OD):
    """Engine for STTN (multivariate-Gaussian OD demand).

    The model returns ``(loc, Sigma)`` where, per (batch, horizon, origin),
    ``loc`` is a mean over the ``N`` destinations and ``Sigma`` is the
    ``N×N`` destination covariance.  This engine trains on the multivariate
    normal NLL (``MGAU``) and reports point metrics on ``loc``.  The OD shape
    handling (``_collect`` / ``_horizon_slice``), epoch loop, checkpointing and
    export are inherited from :class:`base.engine.BaseEngine_OD`.
    """

    @staticmethod
    def _to_event_last(loc, sigma):
        """Reorder so the destination axis is the multivariate-normal event dim.

        ``loc (B,H,No,Nd,Dc)``      -> ``(B,H,No,Dc,Nd)``
        ``sigma (B,H,No,Nd,Nd,Dc)`` -> ``(B,H,No,Dc,Nd,Nd)``
        """
        loc_e = loc.permute(0, 1, 2, 4, 3)
        sig_e = sigma.permute(0, 1, 2, 5, 3, 4)
        return loc_e, sig_e

    def _nll(self, loc, sigma, label, mask_value):
        loc_e, sig_e = self._to_event_last(loc, sigma)
        label_e = label.permute(0, 1, 2, 4, 3)
        return mnormal_loss(loc_e, label_e, mask_value, sig_e)

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

            loc, sigma = self.model(X)
            if self._normalize:
                loc, label = self._inverse_transform(
                    [loc, label], device=self._device.type
                )
            loss = self._nll(loc, sigma, label, mask_value)

            with torch.no_grad():
                self.metric.compute_one_batch(loc, label, mask_value, "train")

            loss.backward()
            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                loc, _ = self.model(X)

                if self._normalize:
                    loc, label = self._inverse_transform(
                        [loc, label], device=self._device.type
                    )

                if mode == "val":
                    self.metric.compute_one_batch(
                        loc, label, self._mask_value.to(loc.device), "valid"
                    )
                else:
                    preds.append(self._collect(loc).cpu())
                    labels.append(self._collect(label).cpu())

        if mode == "val":
            return

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                self.metric.compute_one_batch(
                    self._horizon_slice(preds, i),
                    self._horizon_slice(labels, i),
                    mask_value,
                    "test",
                )

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(preds, labels)
