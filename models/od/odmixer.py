"""ODMixer adapted from https://github.com/KLatitude/ODMixer.

Preserves the released CM/MM/BTL blocks, shared two-branch weights, and both
supervised MAE losses. The project adapter supplies completed OD histories,
independent mobility channels, and a multi-horizon output head.
"""

import torch
from torch import nn
from models.base import BaseODModel
from engine.recipe import ModelRecipe
from engine.trainer import BaseEngine_OD
from data.loader import TimeSeriesDataset, LoaderAdapter, _read
from data.temporal import steps_per_day
from engine.metrics import masked_mae
import numpy as np


class SingleInteractModule(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout=0.1):
        super(SingleInteractModule, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim),
        )
        self.conv1d = nn.Conv1d(2, 1, 1)

    def forward(self, x, y):
        shape = x.shape
        x_to_y, y_to_x = x.reshape(shape[0], -1), y.reshape(shape[0], -1)
        z = torch.cat((x_to_y.unsqueeze(-1), y_to_x.unsqueeze(-1)), -1)
        z = torch.squeeze(self.conv1d(z.permute(0, 2, 1)))
        z = z.reshape(shape)
        gate = torch.sigmoid(self.linear1(z))
        output = self.linear2(x) * gate
        return output + x


class BTL(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.1):
        super().__init__()

        self.up_interact = SingleInteractModule(input_dim, hid_dim, input_dim, dropout)
        self.down_interact = SingleInteractModule(input_dim, hid_dim, input_dim, dropout)

    def forward(self, x, y):
        output_x = self.up_interact(x, y)
        output_y = self.down_interact(y, x)
        return output_x, output_y


class MixerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        y = self.ffn(x)
        return y


class CM(nn.Module):
    def __init__(self, input_seq, input_dim, hid_dim, num_nodes, dropout):
        super().__init__()

        self.input_seq = input_seq
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.mixer = MixerLayer(self.hid_dim, 2 * self.hid_dim, self.hid_dim, self.dropout)
        self.ln = nn.LayerNorm([self.num_nodes, self.input_dim, self.hid_dim])

    def forward(self, x):
        b, n, m, d = x.shape
        feat = self.mixer(x)
        output = self.ln(feat + x)
        return output


class MM(nn.Module):
    def __init__(self, input_seq, input_dim, hid_dim, num_nodes, dropout):
        super().__init__()

        self.input_seq = input_seq
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.origin_mixer = MixerLayer(
            self.num_nodes, 2 * self.hid_dim, self.num_nodes, self.dropout
        )
        self.des_mixer = MixerLayer(self.num_nodes, 2 * self.hid_dim, self.num_nodes, self.dropout)
        self.ln = nn.LayerNorm([self.num_nodes, self.input_dim, self.hid_dim])

    def forward(self, x):
        b, n, m, d = x.shape

        origin_feat = x.permute(0, 1, 3, 2)
        origin_feat = self.origin_mixer(origin_feat)
        origin_feat = origin_feat.permute(0, 1, 3, 2)

        des_feat = x.permute(0, 2, 3, 1)
        des_feat = self.des_mixer(des_feat)
        des_feat = des_feat.permute(0, 3, 1, 2)

        feat = origin_feat + des_feat
        output = self.ln(feat + x)
        return output


class ODIM(nn.Module):
    def __init__(self, input_seq, input_dim, hid_dim, num_nodes, dropout):
        super().__init__()

        self.cm = CM(input_seq, input_dim, hid_dim, num_nodes, dropout)
        self.mm = MM(input_seq, input_dim, hid_dim, num_nodes, dropout)

    def forward(self, x):
        h = self.cm(x)
        output = self.mm(h)
        return output


class ODMixer(BaseODModel):
    def __init__(
        self,
        seq_len,
        input_dim,
        hid_dim,
        node_num,
        layer_nums,
        dropout,
        feature,
        horizon,
        output_dim,
        **args,
    ):
        super(ODMixer, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)
        if hid_dim < 2 or layer_nums < 1 or not 0 <= dropout < 1:
            raise ValueError("ODMixer requires hid_dim >= 2, layer_nums >= 1, and 0 <= dropout < 1")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.node_num = node_num
        self.layer_nums = layer_nums
        self.dropout = dropout

        # OD pair view embedding layer
        self.emb_layer = nn.Linear(self.seq_len, self.hid_dim)

        self.encoder_layer = nn.ModuleList(
            [ODIM(seq_len, input_dim, hid_dim, node_num, dropout) for _ in range(self.layer_nums)]
        )
        self.trend_layer = nn.ModuleList(
            [BTL(self.hid_dim, self.hid_dim, self.dropout) for _ in range(self.layer_nums)]
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.PReLU(),
            nn.Linear(self.hid_dim // 2, horizon),
        )

    def forward(self, X, label=None, return_aux=False):
        if not isinstance(X, dict) or "prev_od" not in X:
            raise ValueError("ODMixer requires current and previous-day windows; use its dataset loader")
        od, batch, channels, squeeze = self._fold_channels(X["od"])
        prev_od, _, _, _ = self._fold_channels(X["prev_od"])
        b, t, n, m = od.shape
        od = od.permute(0, 2, 3, 1)
        od_feat = self.emb_layer(od)
        prev_od_feat = self.emb_layer(prev_od.permute(0, 2, 3, 1))

        for i in range(self.layer_nums):
            od_feat = self.encoder_layer[i](od_feat)
            prev_od_feat = self.encoder_layer[i](prev_od_feat)
            prev_od_feat, od_feat = self.trend_layer[i](prev_od_feat, od_feat)

        od_output = self.output_layer(od_feat)
        od_output = od_output.reshape(b, n, m, -1).permute(0, 3, 1, 2)

        point = self._unfold_channels(od_output, batch, channels, squeeze)
        if not return_aux:
            return point
        prev_output = self.output_layer(prev_od_feat).reshape(b, n, m, -1).permute(0, 3, 1, 2)
        return point, self._unfold_channels(prev_output, batch, channels, squeeze)


class ODMixerDataset(TimeSeriesDataset):
    """Completed OD histories at this origin and the same time yesterday."""

    def __init__(self, data, indices, seq_len, horizon, period):
        indices = np.asarray(indices)
        indices = indices[indices - period >= seq_len - 1]
        if period < horizon:
            raise ValueError("ODMixer auxiliary target must be observed: horizon must not exceed period")
        super().__init__(data, indices, seq_len, horizon)
        self.period = period

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        t = self.indices[i] - self.period
        return {"od": x, "prev_od": torch.from_numpy(self.data[t + self.x_offsets]),
                "prev_y_od": torch.from_numpy(self.data[t + self.y_offsets])}, y


def load_data(data_path, config, logger, drop=False):
    folder, data, scaler = _read(data_path, config)
    period = config.model.params.period or steps_per_day(config.data.frequency)
    loaders = {}
    splits = ["train", "val", "test"]
    if config.data.protocol == "revision":
        splits += ["tune", "fit", "cal"]
    for split in splits:
        indices = np.load(folder / f"idx_{split}.npy")
        usable = indices[indices >= config.data.seq_len - 1 + period]
        if split != "train" and len(usable) != len(indices):
            raise ValueError(f"{split} lacks yesterday's history; common evaluation origins are required")
        limit = getattr(config.runtime, f"max_{split}_samples", None)
        ds = ODMixerDataset(data, usable[:limit], config.data.seq_len, config.data.horizon, period)
        loaders[f"{split}_loader"] = LoaderAdapter(
            ds, config.training.batch_size, shuffle=split == "train", drop_last=drop,
            num_workers=config.runtime.num_workers, pin_memory=config.runtime.pin_memory,
            logger=logger, name=split,
        )
    return loaders, scaler


class ODMixerEngine(BaseEngine_OD):
    """The author implementation's current MAE + previous-day auxiliary MAE."""

    def train_batch(self):
        self.model.train()
        for X, label in self._dataloader["train_loader"].get_iterator():
            X, label = self._prepare_batch([X, label])
            self._optimizer.zero_grad(set_to_none=True)
            point, previous = self.model(X, return_aux=True)
            previous_label = X["prev_y_od"]
            if self._normalize:
                point, previous, label, previous_label = self._inverse_transform(
                    [point, previous, label, previous_label], device=self._device
                )
            mask_value = self._mask_value.to(point.device)
            main_loss = self.metric.compute_one_batch(point, label, mask_value, "train")
            loss = main_loss + masked_mae(previous, previous_label, mask_value)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite ODMixer dual-branch loss")
            loss.backward()
            if self._clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm, error_if_nonfinite=True)
            self._optimizer.step()
            self._iter_cnt += 1


def build_model(config, node_num, **ctx):
    return ODMixer(
        node_num=node_num,
        input_dim=node_num,
        output_dim=config.data.output_dim,
        dropout=config.model.params.dropout,
        feature=config.data.input_dim,
        horizon=config.data.horizon,
        seq_len=config.data.seq_len,
        hid_dim=config.model.params.hid_dim,
        layer_nums=config.model.params.layer_nums,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, loss_fn="MAE", od=True, od_cqr=True,
                       load_data=load_data, engine_cls=ODMixerEngine)
