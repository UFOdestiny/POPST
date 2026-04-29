import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from base.engine import BaseEngine


_SUPPORTED_LOSSES = {"MAE", "MSE", "RMSE", "MAPE", "KL"}


class FlowMatchingEngine(BaseEngine):
    """Flow-matching engine for models exposing point + vector-field hooks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._optimizer is None:
            raise ValueError("FlowMatchingEngine requires an optimizer.")

        if self._loss_fn not in _SUPPORTED_LOSSES:
            supported = ", ".join(sorted(_SUPPORTED_LOSSES))
            raise ValueError(
                f"FlowMatchingEngine does not support loss_fn={self._loss_fn!r}. "
                f"Supported losses: {supported}."
            )

        self.flow_weight = getattr(self.args, "fm_flow_weight", 0.1)
        self.ode_steps = getattr(self.args, "fm_ode_steps", 20)
        self.num_samples = getattr(self.args, "fm_num_samples", 10)
        self.output_activation = getattr(self.args, "fm_output_activation", "relu")
        self._validate_model_hooks()

        self._logger.info(f"{'Engine Mode':20s}: flow_matching")
        self._logger.info(f"{'FM Flow Weight':20s}: {self.flow_weight}")
        self._logger.info(f"{'FM ODE Steps':20s}: {self.ode_steps}")
        self._logger.info(f"{'FM Num Samples':20s}: {self.num_samples}")
        self._logger.info(f"{'FM Output Act':20s}: {self.output_activation}")

    def _validate_model_hooks(self):
        missing = [
            name
            for name in ("forward_point", "forward_flow")
            if not callable(getattr(self.model, name, None))
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise TypeError(
                f"{type(self.model).__name__} cannot run with FlowMatchingEngine. "
                f"Missing required hook(s): {missing_str}."
            )

    def _forward_point(self, x):
        outputs = self.model.forward_point(x)
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise TypeError(
                "model.forward_point(x) must return at least "
                "(point_pred, flow_context...)."
            )
        point_pred = outputs[0]
        flow_context = tuple(outputs[1:])
        return point_pred, flow_context

    def _forward_flow(self, x_t, flow_context, t):
        return self.model.forward_flow(x_t, *flow_context, t)

    def _metric_tensors(self, pred, label):
        if not self._normalize:
            return pred, label
        return self._inverse_transform([pred, label], device=self._device.type)

    @staticmethod
    def _expand_time(t, ref):
        return t.view(t.shape[0], *([1] * (ref.ndim - 1)))

    def _build_flow_targets(self, point_pred, label):
        batch_size = label.shape[0]
        t = torch.rand(batch_size, device=label.device)
        noise = torch.randn_like(label)
        residual_target = label - point_pred.detach()
        t_expand = self._expand_time(t, label)
        x_t = (1.0 - t_expand) * noise + t_expand * residual_target
        target_v = residual_target - noise
        return x_t, target_v, t

    def _apply_output_activation(self, pred):
        if self.output_activation == "relu":
            return F.relu(pred)
        return pred

    def _sample_predictions(self, point_pred, flow_context, shape, steps=None):
        steps = self.ode_steps if steps is None else steps
        batch_size = point_pred.shape[0]
        residual = torch.randn(shape, device=self._device)
        dt = 1.0 / steps

        for step_idx in range(steps):
            t = torch.full(
                (batch_size,),
                step_idx * dt,
                device=self._device,
                dtype=point_pred.dtype,
            )
            residual = residual + self._forward_flow(residual, flow_context, t) * dt

        return self._apply_output_activation(point_pred + residual)

    def _sample_batch_predictions(self, point_pred, flow_context, label_shape):
        batch_samples = []
        for _ in range(self.num_samples):
            pred = self._sample_predictions(point_pred, flow_context, label_shape)
            batch_samples.append(pred)
        return torch.stack(batch_samples, dim=1)

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n"
                    + "=" * 25
                    + "   Training   "
                    + "=" * 25
                )

            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])
            point_pred, flow_context = self._forward_point(X)
            x_t, target_v, t = self._build_flow_targets(point_pred, label)
            pred_v = self._forward_flow(x_t, flow_context, t)

            metric_pred, metric_label = self._metric_tensors(point_pred, label)
            point_loss = self.metric.compute_one_batch(
                metric_pred, metric_label, mask_value, "train"
            )
            (point_loss + self.flow_weight * F.mse_loss(pred_v, target_v)).backward()

            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1

    def train(self):
        wait = 0
        min_loss_val = np.inf

        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()

            v1 = time.time()
            valid_loss = self.evaluate("val")
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(
                epoch + 1, cur_lr, t2 - t1, v2 - v1, 0.0, include_test=False
            )
            self._logger.info(msg)

            if valid_loss < min_loss_val:
                if valid_loss == 0:
                    self._logger.info("Something went WRONG!")
                    exit()

                self.save_model(self._save_path)
                self._logger.info(
                    "Val  loss: {:.3f} -> {:.3f}".format(min_loss_val, valid_loss)
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
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        mask_value = self._mask_value.to(self._device)
        sample_preds, point_preds, labels = [], [], []
        loader_key = "test_loader" if mode == "export" else f"{mode}_loader"

        with torch.no_grad():
            for X, label in self._dataloader[loader_key].get_iterator():
                X, label = self._prepare_batch([X, label])

                if mode == "val":
                    point_pred, flow_context = self._forward_point(X)
                    pred = self._sample_batch_predictions(
                        point_pred, flow_context, label.shape
                    ).mean(dim=1)
                    metric_pred, metric_label = self._metric_tensors(pred, label)
                    self.metric.compute_one_batch(
                        metric_pred, metric_label, mask_value, "valid"
                    )
                    continue

                point_pred, flow_context = self._forward_point(X)
                pred = self._sample_batch_predictions(
                    point_pred, flow_context, label.shape
                )

                if self._normalize:
                    batch_shape = pred.shape
                    pred_flat = pred.reshape(-1, *pred.shape[2:])
                    pred_flat, point_pred, label = self._inverse_transform(
                        [pred_flat, point_pred, label], device=self._device.type
                    )
                    pred = pred_flat.reshape(batch_shape)

                sample_preds.append(pred.cpu())
                point_preds.append(point_pred.cpu())
                labels.append(label.cpu())

        if mode == "val":
            return self.metric.get_valid_loss()

        sample_preds = torch.cat(sample_preds, dim=0)
        point_preds = torch.cat(point_preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mean_preds = sample_preds.mean(dim=1)

        if mode in {"test", "export"}:
            for i in range(self.model.horizon):
                self.metric.compute_one_batch(
                    mean_preds[:, i : i + 1],
                    labels[:, i : i + 1],
                    self._mask_value,
                    "test",
                )

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(sample_preds, point_preds, labels)
                self.save_test()

    def save_result(self, preds, point_preds, labels):
        preds_np = preds.numpy()
        point_preds_np = point_preds.numpy()
        labels_np = labels.numpy()

        base_name = f"{self.args.model_name}-{self.args.dataset}-res"
        save_name = f"{base_name}.npz"
        path = os.path.join(self._save_path, save_name)

        suffix = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{suffix}.npz"
            path = os.path.join(self._save_path, save_name)
            suffix += 1

        np.savez(path, preds=preds_np, point_preds=point_preds_np, labels=labels_np)

        self._logger.info(f"Results Save Path: {path}")
        self._logger.info(
            f"Preds Shape: {preds_np.shape} "
            "(test size, num_samples, horizon, region, channels)"
        )
        self._logger.info(
            f"Point Preds Shape: {point_preds_np.shape} "
            "(test size, horizon, region, channels)"
        )
        self._logger.info(
            f"Labels Shape: {labels_np.shape} "
            "(test size, horizon, region, channels)\n\n"
        )
