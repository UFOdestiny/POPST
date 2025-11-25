import torch
import numpy as np
from base.engine import BaseEngine
import time

class STTN_Engine(BaseEngine):
    def __init__(self, **args):
        super(STTN_Engine, self).__init__(**args)
    
    def train(self):
        wait = 0
        min_loss_val = np.inf
        min_loss_test = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()

            v1 = time.time()
            self.evaluate("val")
            v2 = time.time()

            te1 = time.time()
            # self.evaluate("test", export=None, train_test=True)
            te2 = time.time()

            valid_loss = self.metric.get_valid_loss()
            test_loss = self.metric.get_test_loss()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(
                epoch + 1, cur_lr, t2 - t1, v2 - v1, te2 - te1
            )
            self._logger.info(msg)

            if test_loss < min_loss_test:
                self._logger.info(
                    "Test loss: {:.3f} -> {:.3f}".format(
                        min_loss_test, test_loss
                    )
                )
                min_loss_test = test_loss

            if valid_loss < min_loss_val:
                if valid_loss == 0:
                    self._logger.info("Something went WRONG!")
                    exit()
                    break

                self.save_model(self._save_path)
                self._logger.info(
                    "Val  loss: {:.3f} -> {:.3f}".format(
                        min_loss_val, valid_loss
                    )
                )
                min_loss_val = valid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(
                            epoch + 1, min_loss_val
                        )
                    )
                    break

        self.evaluate("test", export=self.args.export)



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
                pred = self.model(X)
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

        elif mode == "test":
            n = preds.shape[0]//self.args.bs+1
            for i in range(n):
                N=self.args.bs*(i+1)
                N_=self.args.bs*i
                self.metric.compute_one_batch(preds[N_:N,...], labels[N_:N,...], mask_value, "test", scale=scales[N_:N,...])

            if not train_test:
                for i in self.metric.get_test_msg():
                    self._logger.info(i)

            if export:
                self.save_result(preds, labels)

                # # metrics
                # metrics = np.vstack(self.metric.export())
                # np.save(f"{self._save_path}/metrics.npy", metrics)
                # self._logger.info(f'metrics results shape: {metrics.shape} {self.metric.metric_lst})')
