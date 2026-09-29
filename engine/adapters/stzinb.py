import torch
from engine.trainer import BaseEngine_OD
from engine.metrics import zinb_mean, zinb_nll


class STZINB_Engine(BaseEngine_OD):
    """Count-space ZINB training and evaluation for the OD adaptation of STZINB.

    Source: https://github.com/ZhuangDingyi/STZINB.
    Only normalized labels are inverse-transformed; (n,p,pi) already parameterize
    counts. Validation NLL selects checkpoints; point metrics use the ZINB mean."""

    def _to_counts(self, tensor):
        """Inverse-transform a tensor to the original count space when the data
        was normalised; the ZINB likelihood is defined over raw counts."""
        if self._normalize:
            tensor = self._inverse_transform(tensor, device=self._device)
        return tensor

    # -- training -------------------------------------------------------

    def train_batch(self):
        self.model.train()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n" + "=" * 25 + "   Training   " + "=" * 25
                )
            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])

            n, p, pi = self.model(X)
            # ZINB likelihood lives in the original count space; only the
            # label is inverse-transformed (n/p/pi are model outputs fit
            # directly against count-space labels, so zinb_mean(n,p,pi) is
            # already in count space and must not be inverse-transformed
            # again).
            label_c = self._to_counts(label)
            loss = zinb_nll(n, p, pi, label_c, null_val=mask_value)

            # Track point metrics against the ZINB mean for monitoring.
            with torch.no_grad():
                pred = zinb_mean(n, p, pi)
                self.metric.compute_one_batch(pred, label_c, mask_value, "train", value=loss)

            loss.backward()
            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)
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
                # zinb_mean(n, p, pi) is already in count space (n/p/pi are
                # model outputs, not normalised data) — only the label needs
                # inverse-transforming back to count space to match it.
                pred = zinb_mean(n, p, pi)
                label = self._to_counts(label)

                if mode == "val":
                    mask_value = self._mask_value.to(pred.device)
                    nll = zinb_nll(n, p, pi, label, null_val=mask_value)
                    self.metric.compute_one_batch(pred, label, mask_value, "valid", value=nll)
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
                    self._horizon_slice(ns, i),
                    self._horizon_slice(ps, i),
                    self._horizon_slice(pis, i),
                    self._horizon_slice(labels, i),
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
