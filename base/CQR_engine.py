import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.engine import BaseEngine
from base.metrics import Metrics


class QuantileOutputLayer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.feature_dim = feature_dim

        layers = []
        in_dim = feature_dim
        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feature_dim * 3))
        self.net = nn.Sequential(*layers)

    def forward(self, preds: torch.Tensor):
        # preds: (B, T, N, F)
        logits = self.net(preds)
        logits = logits.view(*preds.shape[:-1], self.feature_dim, 3)

        base = logits[..., 1]
        lower = base - F.softplus(logits[..., 0])
        upper = base + F.softplus(logits[..., 2])
        mid = base

        return lower, mid, upper


class CQR_Engine(BaseEngine):
    DEFAULT_METRICS = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "IS", "COV"]

    def __init__(self, quantile_hidden_dim: Optional[int] = None, **args):
        args["loss_fn"] = "Quantile"
        # args["metric_list"] = args.get("metric_list") or self.DEFAULT_METRICS
        args["metric_list"] = self.DEFAULT_METRICS
        super().__init__(**args)

        self.quantile_head = QuantileOutputLayer(
            self.model.output_dim, quantile_hidden_dim
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

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
                torch.nn.utils.clip_grad_norm_(
                    self.quantile_head.parameters(), self._clip_grad_value
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
        save_name = f"{self.args.model_name}-{self.args.dataset}-res.npy"
        path = os.path.join(self._save_path, save_name)
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
