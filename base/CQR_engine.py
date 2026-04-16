import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.engine import BaseEngine
from base.metrics import Metrics


class QuantileOutputLayer(nn.Module):
    """Neural network head that produces prediction intervals (lower, mid, upper).

    Wraps a base model's point predictions and learns uncertainty bounds
    using softplus-activated deltas with configurable minimum width.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        init_width: float = 0.1,
        min_width: float = 1e-3,
        learnable_mid: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.min_width = min_width
        self.learnable_mid = learnable_mid

        # Output channels: 2 (lower_delta, upper_delta) or 3 (+ mid_delta)
        out_channels = 3 if learnable_mid else 2

        layers = []
        in_dim = feature_dim
        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feature_dim * out_channels))
        self.net = nn.Sequential(*layers)

        self._init_weights(init_width)

    def _init_weights(self, init_width: float):
        """Initialize so softplus output starts near init_width."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    # softplus(x) ~ x when x > 0, so bias ~ init_width
                    nn.init.constant_(m.bias, init_width)

    def forward(self, preds: torch.Tensor):
        # preds: (B, T, N, F)
        logits = self.net(preds)

        if self.learnable_mid:
            logits = logits.view(*preds.shape[:-1], self.feature_dim, 3)
            lower_delta = F.softplus(logits[..., 0]) + self.min_width
            mid_delta = torch.tanh(logits[..., 1]) * 0.1
            upper_delta = F.softplus(logits[..., 2]) + self.min_width

            mid = preds + mid_delta * preds.abs().clamp(min=1.0)
            lower = mid - lower_delta
            upper = mid + upper_delta
        else:
            logits = logits.view(*preds.shape[:-1], self.feature_dim, 2)
            lower_delta = F.softplus(logits[..., 0]) + self.min_width
            upper_delta = F.softplus(logits[..., 1]) + self.min_width

            mid = preds  # Keep original prediction unchanged
            lower = mid - lower_delta
            upper = mid + upper_delta

        return lower, mid, upper


class CQR_Engine(BaseEngine):
    """Conformal Quantile Regression engine.

    Extends BaseEngine by adding a learnable quantile output head that wraps
    the base model's point predictions into prediction intervals.
    """

    DEFAULT_METRICS = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "IS", "COV"]

    def __init__(
        self,
        quantile_hidden_dim: Optional[int] = None,
        quantile_init_width: float = 0.1,
        quantile_min_width: float = 1e-3,
        quantile_learnable_mid: bool = False,
        **args
    ):
        """
        Args:
            quantile_hidden_dim: Hidden dimension of quantile head
            quantile_init_width: Initial interval width
            quantile_min_width: Minimum interval width
            quantile_learnable_mid: Whether to fine-tune median (False keeps original)
        """
        args["loss_fn"] = "Quantile"
        args["metric_list"] = self.DEFAULT_METRICS
        super().__init__(**args)

        self.quantile_head = QuantileOutputLayer(
            self.model.output_dim,
            hidden_dim=quantile_hidden_dim,
            init_width=quantile_init_width,
            min_width=quantile_min_width,
            learnable_mid=quantile_learnable_mid,
        )
        self.quantile_head.to(self._device)
        self._optimizer.add_param_group({"params": self.quantile_head.parameters()})

        self.metric = Metrics(
            self._loss_fn, args["metric_list"], horizon=self.model.horizon
        )

    def _forward_quantiles(self, X, label):
        pred = self._predict(X, label=label, iter=self._iter_cnt)
        scale = None
        if isinstance(pred, tuple):
            pred, scale = pred
        if self._normalize:
            pred, label = self._inverse_transform(
                [pred, label], device=self._device.type
            )
        lower, mid, upper = self.quantile_head(pred)
        return lower, mid, upper, label, scale

    def train_batch(self):
        self.model.train()
        self.quantile_head.train()
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

            lower, mid, upper, label, scale = self._forward_quantiles(X, label)

            res = self.metric.compute_one_batch(
                mid,
                label,
                mask_value,
                "train",
                lower=lower,
                upper=upper,
                scale=scale,
            )
            res.backward()

            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.quantile_head.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        self.quantile_head.eval()

        mids, lowers, uppers, labels = [], [], [], []

        with torch.no_grad():
            loader_key = "test_loader" if mode == "export" else f"{mode}_loader"
            for X, label in self._dataloader[loader_key].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                lower, mid, upper, label, _ = self._forward_quantiles(X, label)

                mids.append(mid.cpu())
                lowers.append(lower.cpu())
                uppers.append(upper.cpu())
                labels.append(label.cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(torch.nan)

        def _compute(mode_name):
            for h in range(mids.shape[1]):
                self.metric.compute_one_batch(
                    mids[:, h : h + 1],
                    labels[:, h : h + 1],
                    mask_value,
                    mode_name,
                    lower=lowers[:, h : h + 1],
                    upper=uppers[:, h : h + 1],
                )

        if mode == "val":
            _compute("valid")
            return

        if mode in {"test", "export"}:
            _compute("test")

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(mids, lowers, uppers, labels)

    def save_result(self, mids, lowers, uppers, labels):
        result = torch.stack([lowers, mids, uppers, labels], dim=0)
        base_name = f"{self.args.model_name}-{self.args.dataset}-res"
        save_name = f"{base_name}.npy"
        path = os.path.join(self._save_path, save_name)

        # Append numeric suffix if file already exists
        suffix = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{suffix}.npy"
            path = os.path.join(self._save_path, save_name)
            suffix += 1

        np.save(path, result.numpy())
        self._logger.info(
            f"Results Save Path: {path} | Shape: {result.shape} (component, batch, horizon, node, feature)"
        )

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = self._time_model
        state = {
            "model": self.model.state_dict(),
            "quantile_head": self.quantile_head.state_dict(),
        }
        torch.save(state, os.path.join(save_path, filename))

    def _load_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.quantile_head.load_state_dict(
                checkpoint["quantile_head"], strict=False
            )
        else:
            self.model.load_state_dict(checkpoint, strict=False)

    def load_model(self, save_path):
        filename = self._time_model
        f = os.path.join(save_path, filename)
        if not os.path.exists(f):
            models = [i for i in os.listdir(save_path) if i.endswith(".pt")]
            if not models:
                self._logger.info(f"Model {f} Not Exist. No More Models.")
                exit()
            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            f = os.path.join(save_path, models[-1])
            self._logger.info(
                f"Model {filename} Not Exist. Try the Newest Model {os.path.basename(f)}."
            )
        checkpoint = torch.load(f, weights_only=False, map_location=self._device)
        self._load_state_dict(checkpoint)

    def load_exact_model(self, path):
        checkpoint = torch.load(path, weights_only=False, map_location=self._device)
        self._load_state_dict(checkpoint)
