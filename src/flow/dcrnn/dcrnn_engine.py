import torch
import numpy as np
from base.engine import BaseEngine
from base.metrics import masked_mape, masked_rmse, masked_mae, masked_crps, masked_mpiw_ens, masked_kl


class DCRNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DCRNN_Engine, self).__init__(**args)

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_kl = []
        train_mpiw = []
        train_crps = []

        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))

            pred = self.model(X, label, self._iter_cnt)
            if self._normalize:
                pred, label = self._inverse_transform([pred, label])


            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info(f'check mask value {mask_value}')

            loss = self._loss_fn(pred, label, mask_value)

            mae = masked_mae(pred, label, mask_value).item()
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            crps = masked_crps(pred, label, mask_value).item()
            mpiw = masked_mpiw_ens(pred, label, mask_value).item()
            kl = masked_kl(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())

            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)

            train_kl.append(kl)
            train_mpiw.append(mpiw)
            train_crps.append(crps)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse), np.mean(
            train_kl), np.mean(train_mpiw), np.mean(train_crps)

