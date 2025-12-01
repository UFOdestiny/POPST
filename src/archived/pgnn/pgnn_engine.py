from base.engine import BaseEngine

import numpy as np
import torch

from base.CQR_engine import CQR_Engine
from base.metrics import compute_all_metrics, masked_mae, masked_kl, masked_crps, masked_mpiw_ens, masked_mpiw
from base.metrics import masked_mape
from base.metrics import masked_rmse


class PGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(PGNN_Engine, self).__init__(**args)
        self.normalize = args["normalize"]

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

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, X)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(torch.nan)

            if self._iter_cnt == 0:
                self._logger.info(f'check mask value {mask_value}')

            if type(pred) == tuple:
                pred, _ = pred
            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

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


    def evaluate(self, mode, model_path=None, export=None):
        if mode == 'test' or mode == 'export':
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, X)

                if type(pred) == tuple:
                    pred, _ = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(torch.nan)


        if mode == 'val':
            mae = masked_mae(pred, label, mask_value).item()
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            crps = masked_crps(pred, label, mask_value).item()
            mpiw = masked_mpiw_ens(pred, label, mask_value).item()
            kl = masked_kl(pred, label, mask_value).item()
            return mae, mape, rmse, kl, mpiw, crps

        elif mode == 'test' or mode == 'export':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_kl = []
            test_mpiw = []
            test_crps = []

            self._logger.info(f'check mask value {mask_value}')
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)

                log = ('Test Horizon: {:d}, '
                       'MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}')

                self._logger.info(log.format(i + 1, res[0], res[1], res[2], res[3], res[4], res[5]))
                test_mae.append(res[0])
                test_rmse.append(res[1])
                test_mape.append(res[2])
                test_kl.append(res[3])
                test_mpiw.append(res[4])
                test_crps.append(res[5])

            log = 'Average Test MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape),
                                         np.mean(test_kl), np.mean(test_mpiw), np.mean(test_crps)))

            if mode == 'export':
                mae = torch.mean(test_mae[0].unsqueeze(0), axis=1)
                mape = torch.mean(test_mape[0].unsqueeze(0), axis=1)
                rmse = torch.mean(test_rmse[0].unsqueeze(0), axis=1)
                kl = torch.mean(test_kl[0].unsqueeze(0), axis=1)
                mpiw = torch.mean(test_mpiw[0].unsqueeze(0), axis=1)
                crps = torch.from_numpy(test_crps[0])
                crps = torch.mean(crps.unsqueeze(0), axis=1)

                metrics = np.vstack((mae, mape, rmse, kl, mpiw, crps))
                print(metrics.shape)
                np.save(f"{self._save_path}/metrics.npy", metrics)

                preds.squeeze_(dim=1)
                labels.squeeze_(dim=1)
                preds.unsqueeze_(dim=0)
                labels.unsqueeze_(dim=0)

                result = np.vstack((preds, labels))
                np.save(f"{self._save_path}/preds_labels.npy", result)




class PGNN_Engine_Quantile(CQR_Engine):
    def __init__(self, **args):
        super(PGNN_Engine_Quantile, self).__init__(**args)
        self.normalize = args["normalize"]

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

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, X)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(torch.nan)

            if self._iter_cnt == 0:
                self._logger.info(f'check mask value {mask_value}')

            if type(pred) == tuple:
                pred, _ = pred
            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            mid= torch.unsqueeze(pred[:,0,:,:],1)
            lower=torch.unsqueeze(pred[:,1,:,:],1)
            upper=torch.unsqueeze(pred[:,2,:,:],1)

            loss = self._loss_fn(mid, label, mask_value)
            lower_loss = torch.mean(torch.max((self.lower_bound - 1) * (label - lower), self.lower_bound * (label - lower)))
            upper_loss = torch.mean(torch.max((self.upper_bound - 1) * (label - upper), self.upper_bound * (label - upper)))
            loss = loss + lower_loss + upper_loss

            mae = masked_mae(mid, label, mask_value).item()
            mape = masked_mape(mid, label, mask_value).item()
            rmse = masked_rmse(mid, label, mask_value).item()
            crps = masked_crps(mid, label, mask_value).item()
            mpiw = masked_mpiw(lower, upper, mask_value).item()
            kl = masked_kl(mid, label, mask_value).item()

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


    def evaluate(self, mode, model_path=None, export=None):
        if mode == 'test' or mode == 'export':
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        mids=[]
        lowers=[]
        uppers=[]
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, X)

                if type(pred) == tuple:
                    pred, _ = pred

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


        if mode == 'val':
            mae = masked_mae(mid, label, mask_value).item()
            mape = masked_mape(mid, label, mask_value).item()
            rmse = masked_rmse(mid, label, mask_value).item()
            crps = masked_crps(mid, label, mask_value).item()
            mpiw = masked_mpiw(lower, upper, mask_value).item()
            kl = masked_kl(mid, label, mask_value).item()
            return mae, mape, rmse, kl, mpiw, crps

        elif mode == 'test' or mode == 'export':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_kl = []
            test_mpiw = []
            test_crps = []

            self._logger.info(f'check mask value {mask_value}')
            for i in range(self.model.horizon):
                res = compute_all_metrics(mids[:, i, :], labels[:, i, :], mask_value)

                log = ('Test Horizon: {:d}, '
                       'MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}')

                self._logger.info(log.format(i + 1, res[0], res[1], res[2], res[3], res[4], res[5]))
                test_mae.append(res[0])
                test_rmse.append(res[1])
                test_mape.append(res[2])
                test_kl.append(res[3])
                test_mpiw.append(res[4])
                test_crps.append(res[5])

            log = 'Average Test MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape),
                                         np.mean(test_kl), np.mean(test_mpiw), np.mean(test_crps)))

            if mode == 'export':
                mae = torch.mean(test_mae[0].unsqueeze(0), axis=1)
                mape = torch.mean(test_mape[0].unsqueeze(0), axis=1)
                rmse = torch.mean(test_rmse[0].unsqueeze(0), axis=1)
                kl = torch.mean(test_kl[0].unsqueeze(0), axis=1)
                mpiw = torch.mean(test_mpiw[0].unsqueeze(0), axis=1)
                crps = torch.from_numpy(test_crps[0])
                crps = torch.mean(crps.unsqueeze(0), axis=1)

                metrics = np.vstack((mae, mape, rmse, kl, mpiw, crps))
                print(metrics.shape)
                np.save(f"{self._save_path}/metrics.npy", metrics)

                mids.squeeze_(dim=1)
                lowers.squeeze_(dim=1)
                uppers.squeeze_(dim=1)
                labels.squeeze_(dim=1)

                mids.unsqueeze_(dim=0)
                lowers.squeeze_(dim=0)
                uppers.squeeze_(dim=0)
                labels.unsqueeze_(dim=0)

                result = np.vstack((mids, lowers, uppers, labels))
                np.save(f"{self._save_path}/preds_labels.npy", result)