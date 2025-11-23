import torch
import numpy as np
from base.engine import BaseEngine
from base.quantile_engine import Quantile_Engine
from base.metrics import (
    masked_mape,
    masked_rmse,
    masked_mae,
    masked_crps,
    masked_mpiw_ens,
    masked_kl,
)


class DCRNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DCRNN_Engine, self).__init__(**args)

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        for X, label in self._dataloader["train_loader"].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label, self._iter_cnt)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(torch.nan)

            if self._iter_cnt == 0:
                self._logger.info(f"check mask value {mask_value}")

            scale = None
            if type(pred) == tuple:
                pred, scale = pred  # mean scale

            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            res = self.metric.compute_one_batch(
                pred, label, mask_value, "train", scale=scale
            )
            res.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()

            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and train_test == False:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds = []
        labels = []
        scales = []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label, self._iter_cnt)
                scale = None
                if type(pred) == tuple:
                    pred, scale = pred  # mean scale

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
                if scale is not None:
                    scales.append(scale.squeeze(-1).cpu())
        if scales:
            scales = torch.cat(scales, dim=0)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(torch.nan)

        if mode == "val":
            self.metric.compute_one_batch(pred, label, mask_value, "valid", scale=scale)

        elif mode == "test" or mode == "export":
            for i in range(self.model.horizon):
                s = scales[:, i, :].unsqueeze(1) if len(scales) > 0 else None
                self.metric.compute_one_batch(
                    preds[:, i, :].unsqueeze(1),
                    labels[:, i, :].unsqueeze(1),
                    mask_value,
                    "test",
                    scale=s,
                )

            if not train_test:
                for i in self.metric.get_test_msg():
                    self._logger.info(i)

            if export:
                self.save_result(preds, labels)

class DCRNN_Engine_Quantile(Quantile_Engine):
    def __init__(self, **args):
        super(DCRNN_Engine_Quantile, self).__init__(**args)

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        for X, label in self._dataloader["train_loader"].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label, self._iter_cnt)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(torch.nan)

            if self._iter_cnt == 0:
                self._logger.info(f"check mask value {mask_value}")

            scale = None
            if type(pred) == tuple:
                pred, scale = pred  # mean scale

            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            # print(pred.shape)

            mid = torch.unsqueeze(pred[:, 0, :, :], 1)
            lower = torch.unsqueeze(pred[:, 1, :, :], 1)
            upper = torch.unsqueeze(pred[:, 2, :, :], 1)

            res = self.metric.compute_one_batch(
                mid, label, mask_value, "train", upper=upper, lower=lower
            )
            res.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()

            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and train_test == False:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        mids = []
        lowers = []
        uppers = []

        labels = []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label, self._iter_cnt)
                scale = None
                if type(pred) == tuple:
                    pred, scale = pred  # mean scale

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                mid = torch.unsqueeze(pred[:, 0, :, :], 1)
                lower = torch.unsqueeze(pred[:, 1, :, :], 1)
                upper = torch.unsqueeze(pred[:, 2, :, :], 1)

                mids.append(mid.squeeze(-1).cpu())
                lowers.append(lower.squeeze(-1).cpu())
                uppers.append(upper.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(torch.nan)

        if mode == "val":
            self.metric.compute_one_batch(
                mids, labels, mask_value, "valid", upper=uppers, lower=lowers
            )

        elif mode == "test" or mode == "export":
            for i in range(self.model.horizon):
                self.metric.compute_one_batch(
                    mids,
                    labels,
                    mask_value,
                    "test",
                    upper=uppers,
                    lower=lowers,
                )

            if not train_test:
                for i in self.metric.get_test_msg():
                    self._logger.info(i)

            if export:
                self.save_result(mids, uppers, lowers, labels)
