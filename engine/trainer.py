import os
import time
import json
import numpy as np
import torch
from engine.metrics import Metrics


class BaseEngine:
    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_norm,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
        config,
        normalize=True,
        metric_list=None,
        init_weights=False,
        min_delta=0.001,
        **kwargs,
    ):
        super().__init__()

        self._normalize = normalize
        self._device = device
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_norm = clip_grad_norm
        self._max_epochs = max_epochs
        self._patience = patience
        self._min_delta = min_delta
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self.config = config
        self._mask_value = torch.tensor(float("nan"))

        self.model = model
        self.model.to(self._device)

        # Optional Xavier weight initialization (replaces AGCRN/ASTGCN/DSTAGNN engines)
        if init_weights:
            self._init_model_weights()

        # Metrics
        if metric_list is None:
            metric_list = ["MAE", "MAPE", "MSE", "RMSE", "KL", "CRPS"]
        self.metric = Metrics(self._loss_fn, metric_list, self.model.horizon)

        self._logger.info(f"{'Loss Function':20s}: {self._loss_fn}")
        self._logger.info(f"{'Parameters':20s}: {self.model.param_num()}")

        self._time_model = "best.pt"

        self._logger.info(f"Model Save Path: {os.path.join(self._save_path, self._time_model)}")

    def _init_model_weights(self):
        """Xavier/uniform initialization for model parameters."""
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def _predict(self, x, label, iter, *args):
        return self.model(x)

    def _collect(self, t):
        """Post-process a prediction/label tensor before stacking over the test
        set.  Flow models drop the trailing singleton feature axis so tensors
        are ``(B, T, N)``; OD engines override this to keep the OD-matrix axes.
        """
        return t.squeeze(-1)

    def _horizon_slice(self, t, i):
        """Slice horizon step *i* from a stacked tensor, keeping a length-1 time
        axis at position 1 so the metric sees ``(B, 1, ...)``.  Default assumes
        the flow layout ``(B, T, N)``; OD engines override for ``(B, T, N, N, D)``.
        """
        return t[:, i, :].unsqueeze(1)

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [self._to_device(t) for t in tensors]
        if isinstance(tensors, dict):
            return {k: self._to_device(v) for k, v in tensors.items()}
        return tensors.to(self._device)

    def _prepare_batch(self, batch):
        return self._to_device(self._to_tensor(batch))

    def _to_tensor(self, nparray):
        if isinstance(nparray, dict):
            return {k: self._to_tensor(v) for k, v in nparray.items()}
        if isinstance(nparray, list):
            return [self._to_tensor(arr) for arr in nparray]
        if isinstance(nparray, torch.Tensor):
            return nparray
        return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors, device="cuda"):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor, device=device)

        if isinstance(tensors, list):
            res = []
            for t in tensors:
                if isinstance(t, tuple):
                    res.append([inv(j) for j in t])
                else:
                    res.append(inv(t))
            return res
        return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = self._time_model
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        self.load_exact_model(os.path.join(save_path, self._time_model))

    def load_exact_model(self, path):
        state = torch.load(path, map_location=self._device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)

    def train_batch(self):
        self.model.train()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n" + "=" * 25 + "   Training   " + "=" * 25
                )
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, f)
            X, label = self._prepare_batch([X, label])
            pred = self._predict(X, label=label, iter=self._iter_cnt)

            scale = None
            if isinstance(pred, tuple):
                pred, scale = pred

            if self._normalize:
                pred, label = self._inverse_transform([pred, label], device=self._device)

            res = self.metric.compute_one_batch(pred, label, mask_value, "train", scale=scale)

            # A single non-finite loss (e.g. an SSM gradient blow-up late in
            # training) would otherwise poison every weight: clip_grad_norm_
            # turns a NaN gradient into a NaN scale, so clipping cannot save it
            # and optimizer.step() spreads the NaN, after which all subsequent
            # predictions are NaN — the masked metrics report 0.000 and the
            # train loop aborts with "Something went WRONG!".  Skip the step.
            if not torch.isfinite(res):
                self._optimizer.zero_grad()
                self._iter_cnt += 1
                continue

            res.backward()

            if self._clip_grad_norm != 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
                # Non-finite gradients can arise even from a finite loss; a NaN
                # norm makes clipping a no-op, so skip the step rather than
                # corrupt the weights (see the loss guard above).
                if not torch.isfinite(total_norm):
                    self._optimizer.zero_grad()
                    self._iter_cnt += 1
                    continue
            self._optimizer.step()
            self._iter_cnt += 1

    def train(self):
        wait = 0
        min_loss_val = np.inf
        best_epoch = None
        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()

            v1 = time.time()
            self.evaluate("val")
            v2 = time.time()

            valid_loss = self.metric.get_valid_loss()
            if not np.isfinite(valid_loss):
                raise RuntimeError("Non-finite validation loss")
            validation = {
                name: float(np.average(values, weights=self.metric.valid_weights))
                for name, values in zip(self.metric.metric_lst, self.metric.valid_res)
            }

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(
                epoch + 1, cur_lr, t2 - t1, v2 - v1
            )
            self._logger.info(msg)

            # Keep the best checkpoint even when its improvement is too small
            # to reset the early-stopping counter.
            improved = valid_loss < min_loss_val
            significant_improvement = improved and valid_loss <= min_loss_val - self._min_delta
            if improved:
                self.save_model(self._save_path)
                self._logger.info("Val  loss: {:.3f} -> {:.3f}".format(min_loss_val, valid_loss))
                min_loss_val = valid_loss
                best_epoch = epoch + 1
                self.last_validation = {
                    "selection_loss": self.metric.loss_name,
                    "best_epoch": best_epoch,
                    "validation": validation,
                }
                with open(os.path.join(self._save_path, "selection.json"), "w", encoding="utf-8") as f:
                    json.dump(self.last_validation, f, indent=2, allow_nan=False)
            if significant_improvement:
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(epoch + 1, min_loss_val)
                    )
                    break

        if best_epoch is None:
            raise RuntimeError("Training produced no finite checkpoint")
        if self.config.training.evaluate_test:
            self.evaluate("test", export=self.config.output.predictions)

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds, labels, scales = [], [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                pred = self._predict(X, label=label, iter=self._iter_cnt)
                scale = None
                if isinstance(pred, tuple):
                    pred, scale = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label], device=self._device)

                if mode == "val":
                    self.metric.compute_one_batch(
                        pred,
                        label,
                        self._mask_value.to(pred.device),
                        "valid",
                        scale=scale,
                    )
                else:
                    preds.append(self._collect(pred).cpu())
                    labels.append(self._collect(label).cpu())
                    if scale is not None:
                        scales.append(self._collect(scale).cpu())

        if mode == "val":
            return

        scales = torch.cat(scales, dim=0) if scales else None
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                s = self._horizon_slice(scales, i) if scales is not None else None
                self.metric.compute_one_batch(
                    self._horizon_slice(preds, i),
                    self._horizon_slice(labels, i),
                    mask_value,
                    "test",
                    scale=s,
                )

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(preds, labels)
                if self.config.output.test_inputs:
                    self.save_test()

    def save_result(self, preds, labels):
        # preds: (B, T, N) or (B, T, N, F)
        # labels: (B, T, N) or (B, T, N, F)
        preds = preds.unsqueeze(0)
        labels = labels.unsqueeze(0)
        # -> (2, B, T, N, ...)
        result = torch.cat([preds, labels], dim=0)
        result_np = result.cpu().numpy()

        path = self._get_unique_save_path("res")

        np.save(path, result_np)

        self._logger.info(f"Results Save Path: {path}")
        self._logger.info(
            f"Results Shape: {result_np.shape} (preds/labels, test size, horizon, region, channels)\n\n"
        )

    def _get_unique_save_path(self, suffix):
        base_name = f"{self.config.model.name}-{self.config.data.id}-{suffix}"
        save_name = f"{base_name}.npy"
        path = os.path.join(self._save_path, save_name)

        index = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{index}.npy"
            path = os.path.join(self._save_path, save_name)
            index += 1

        return path

    def save_test(self):
        test_data = []

        with torch.no_grad():
            for X, _ in self._dataloader["test_loader"].get_iterator():
                if isinstance(X, dict):
                    X = X["od"]
                t = X if isinstance(X, torch.Tensor) else torch.from_numpy(X)
                test_data.append(t.cpu())

        if not test_data:
            self._logger.info("Test data is empty. Skip saving test inputs.")
            return

        test_tensor = torch.cat(test_data, dim=0)
        path = self._get_unique_save_path("test")
        test_np = test_tensor.numpy()
        np.save(path, test_np)

        self._logger.info(f"Test Save Path: {path}")
        self._logger.info(
            f"Test Shape: {test_np.shape} (total test size, seq_len, region, feature)\n\n"
        )


class BaseEngine_OD(BaseEngine):
    """Engine for origin-destination models (see :class:`models.base.BaseODModel`).

    OD predictions and labels are 5-D ``(B, horizon, N, N, D)`` — an ``N×N``
    matrix with ``D`` mobility channels per forecast step — whereas the flow
    pipeline assumes ``(B, T, N, feature=1)`` and drops the trailing axis.
    ``BaseEngine`` already routes the two shape-dependent steps through
    :meth:`_collect` and :meth:`_horizon_slice`; this subclass overrides only
    those, so all of the training loop, checkpointing, logging, metric tracking
    and export are inherited unchanged.
    """

    def _collect(self, t):
        # Keep the full OD-matrix layout (B, H, N, N, D); do not squeeze.
        return t

    def _horizon_slice(self, t, i):
        # (B, H, N, N, D) -> (B, 1, N, N, D) for horizon step i.
        return t[:, i : i + 1]
