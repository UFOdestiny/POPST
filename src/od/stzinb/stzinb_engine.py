import torch

from base.engine import BaseEngine_OD
from base.metrics import zinb_nll, zinb_mean


class STZINB_Engine(BaseEngine_OD):
    """Engine for STZINB (zero-inflated negative binomial OD demand).

    The model returns the distribution parameters ``(n, p, pi)`` rather than a
    point tensor.  This engine:

      * trains on the ZINB negative log-likelihood (in the original count space,
        so labels are inverse-transformed first);
      * reports the standard point metrics (MAE / RMSE / MAPE) against the ZINB
        mean ``E[y] = (1-pi)·n·(1-p)/p``.

    Everything else (epoch loop, checkpointing, per-horizon test aggregation,
    export) is inherited from :class:`base.engine.BaseEngine_OD`; only the two
    shape/loss-specific steps are overridden.
    """

    def _mean_from_params(self, params):
        n, p, pi = params
        return zinb_mean(n, p, pi)

    def _to_counts(self, tensor):
        """Inverse-transform a tensor to the original count space when the data
        was normalised; round non-negative counts for the NB likelihood."""
        if self._normalize:
            tensor = self._inverse_transform(tensor, device=self._device.type)
        return tensor

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

            n, p, pi = self.model(X)
            # ZINB likelihood lives in the original count space.
            n_c = self._to_counts(n)
            label_c = self._to_counts(label)
            loss = zinb_nll(n_c, p, pi, label_c, null_val=mask_value)

            # Track point metrics against the ZINB mean for monitoring.
            with torch.no_grad():
                pred = self._to_counts(self._mean_from_params((n, p, pi)))
                self.metric.compute_one_batch(pred, label_c, mask_value, "train")

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
                params = self.model(X)
                pred = self._mean_from_params(params)

                if self._normalize:
                    pred, label = self._inverse_transform(
                        [pred, label], device=self._device.type
                    )

                if mode == "val":
                    self.metric.compute_one_batch(
                        pred, label, self._mask_value.to(pred.device), "valid"
                    )
                else:
                    preds.append(self._collect(pred).cpu())
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
