import torch
import numpy as np
from base.engine import BaseEngine
import time


class HA_Engine(BaseEngine):
    def __init__(self, **args):
        super(HA_Engine, self).__init__(**args)

    def train_batch(self):
        self.model.train()
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            # print(X.shape, label.shape)
            # self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self._predict(X)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info(f'check mask value {mask_value}')

            scale = None
            if type(pred) == tuple:
                pred, scale = pred  # mean scale

            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            self.metric.compute_one_batch(pred, label, mask_value, "train", scale=scale)

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            # self._optimizer.step()

            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None):
        self.model.eval()

        preds = []
        labels = []
        scales = []

        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                # print(X.shape)

                scale = None
                if type(pred) == tuple:
                    pred, scale = pred  # mean scale

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
                if scale:
                    scales.append(scale.squeeze(-1).cpu())
        if scales:
            scales = torch.cat(scales, dim=0)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            self.metric.compute_one_batch(pred, label, mask_value, "valid", scale=scale)

        elif mode == 'test':
            self._logger.info(f'check mask value {mask_value}')

            for i in range(self.model.horizon):
                s = scales[:, i, :] if len(scales) > 0 else None
                self.metric.compute_one_batch(preds[:, i, :], labels[:, i, :], mask_value, "test", scale=s)

            for i in self.metric.get_test_msg():
                self._logger.info(i)

            if export:
                self.save_result(preds,labels)
                
