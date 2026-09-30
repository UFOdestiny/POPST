"""Occurrence-conditioned event pooling and noncrossing positive quantiles.

The marginal median queries the positive quantile at 1 - 0.5 / P(Y > 0).
Multiplying a conditional median by occurrence probability is not that decision.
The quantile branch learns only from positive marks; the occurrence branch uses
proper, unweighted binary cross entropy. All point losses use count space.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.loader import _read, TimeSeriesDataset, LoaderAdapter
from data.temporal import training_end, steps_per_day
from models.base import BaseODModel
from engine.trainer import BaseEngine_OD
from engine.metrics import masked_mae
from engine.recipe import ModelRecipe


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


class _HurdleBackbone(BaseODModel):
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
        super().__init__(node_num, input_dim, output_dim, seq_len, horizon)
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



def positive_quantile(knots, levels, tau):
    """Linear interpolation with constant extension at both ends."""
    tau = tau.clamp(levels[0], levels[-1])
    upper = torch.searchsorted(levels, tau.contiguous()).clamp(1, len(levels) - 1)
    lower = upper - 1
    lo = torch.gather(knots, -1, lower.unsqueeze(-1)).squeeze(-1)
    hi = torch.gather(knots, -1, upper.unsqueeze(-1)).squeeze(-1)
    alpha = (tau - levels[lower]) / (levels[upper] - levels[lower])
    return lo + alpha * (hi - lo)


def hurdle_median(probability, knots, levels):
    """Bayes MAE decision, with a detached occurrence decision during training."""
    q = probability.detach()
    tau = 1 - .5 / q.clamp_min(1e-6)
    amount = positive_quantile(knots, levels, tau)
    return torch.where(q > .5, amount, torch.zeros_like(amount))


def positive_pinball(knots, target, levels):
    """Conditional positive-mark risk; empty batches retain a zero gradient."""
    active = torch.isfinite(target) & (target > .5)
    if not active.any():
        return knots.sum() * 0
    residual = target[active].unsqueeze(-1) - knots[active]
    return torch.maximum(levels * residual, (levels - 1) * residual).mean()


class PDRHurdleQuantile(_HurdleBackbone):
    def __init__(self, *args, activity_prior, data_min, data_span, use_log1p,
                 quantile_count=11, event_normalization=True, occurrence_weight=.1,
                 quantile_weight=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if quantile_count < 3 or occurrence_weight <= 0 or quantile_weight <= 0:
            raise ValueError('Invalid hurdle quantile configuration')
        self.quantile_count = int(quantile_count)
        if not event_normalization:
            raise ValueError('PDRHurdleQuantile uses active-edge event normalization')
        self.event_normalization = True
        self.occurrence_weight = float(occurrence_weight)
        self.quantile_weight = float(quantile_weight)
        self.use_log1p = bool(use_log1p)
        self.register_buffer('activity_prior', torch.as_tensor(activity_prior, dtype=torch.float32).clamp(1e-5, 1 - 1e-5))
        self.register_buffer('data_min', torch.as_tensor(data_min, dtype=torch.float32).reshape(-1))
        self.register_buffer('data_span', torch.as_tensor(data_span, dtype=torch.float32).reshape(-1))
        self.register_buffer('levels', torch.linspace(.01, .99, quantile_count))
        # Occupancy and conditional mark magnitude remain distinct channels.
        self.origin_events = nn.Sequential(nn.Linear(2 * self.seq_len, self.hid_dim), nn.SiLU())
        self.destination_events = nn.Sequential(nn.Linear(2 * self.seq_len, self.hid_dim), nn.SiLU())
        self.event_norm = nn.LayerNorm(self.hid_dim)
        self.occurrence = nn.Sequential(nn.Linear(self.hid_dim + 8, 32), nn.SiLU(), nn.Linear(32, self.horizon))
        self.positive = nn.Sequential(nn.Linear(self.hid_dim + 8, 32), nn.SiLU(), nn.Linear(32, self.horizon * quantile_count))
        nn.init.zeros_(self.occurrence[-1].weight)
        nn.init.zeros_(self.occurrence[-1].bias)
        nn.init.constant_(self.positive[-1].bias, -2.)
        # The inherited point head is not part of this distributional decoder.
        del self.output_layer

    def _to_counts(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        z = x * self.data_span + self.data_min
        return torch.expm1(z).clamp_min(0) if self.use_log1p else z.clamp_min(0)

    def _to_normalized(self, count, squeeze):
        value = torch.log1p(count) if self.use_log1p else count
        value = (value - self.data_min) / self.data_span
        return value.squeeze(-1) if squeeze else value

    def _context(self, hidden, count, prior):
        # count: (B*channels, O, D, T), with no future observations.
        active = count > .5
        row_n = active.sum(2).to(count.dtype)
        col_n = active.sum(1).to(count.dtype)
        row_den = row_n.clamp_min(1)
        col_den = col_n.clamp_min(1)
        row = torch.cat([row_n / self.node_num, torch.log1p(count.sum(2) / row_den)], -1)
        col = torch.cat([col_n / self.node_num, torch.log1p(count.sum(1) / col_den)], -1)
        h = self.event_norm(hidden + self.origin_events(row).unsqueeze(2) + self.destination_events(col).unsqueeze(1))
        age = (active.flip(-1).float().cumsum(-1) == 0).sum(-1) / self.seq_len
        marks = torch.log1p(count)
        features = torch.stack([active.float().mean(-1), active[..., -1].float(), age,
                                marks.mean(-1), marks[..., -1], marks.std(-1, correction=0),
                                row_n[..., -1].unsqueeze(2).expand_as(age) / self.node_num,
                                prior / 12], -1)
        context = torch.cat([h, features], -1)
        logits = self.occurrence(context) + prior.unsqueeze(-1)
        raw = self.positive(context).reshape(*context.shape[:-1], self.horizon, self.quantile_count)
        # Nonnegative increments enforce noncrossing count-space quantiles.
        knots = 1 + torch.cumsum(F.softplus(raw), -1)
        return logits, knots

    def forward(self, X, label=None, return_aux=False):
        current, batch, channels, squeeze = self._fold_channels(X['od'])
        previous, _, _, _ = self._fold_channels(X['prev_od'])
        h = self.emb_layer(current.permute(0, 2, 3, 1))
        hp = self.emb_layer(previous.permute(0, 2, 3, 1))
        for encoder, trend in zip(self.encoder_layer, self.trend_layer):
            h, hp = encoder(h), encoder(hp)
            hp, h = trend(hp, h)
        counts, _, _, _ = self._fold_channels(self._to_counts(X['od']))
        past_counts, _, _, _ = self._fold_channels(self._to_counts(X['prev_od']))
        prior = self.activity_prior.permute(2, 0, 1)
        if channels != len(prior):
            raise ValueError('Input channels differ from the training prior')
        prior = torch.logit(prior).repeat(batch, 1, 1)
        logits, knots = self._context(h, counts.permute(0, 2, 3, 1), prior)
        past_logits, past_knots = self._context(hp, past_counts.permute(0, 2, 3, 1), prior)
        point = hurdle_median(logits.sigmoid(), knots, self.levels)
        previous_point = hurdle_median(past_logits.sigmoid(), past_knots, self.levels)

        def unfold(value):
            return self._unfold_channels(value.permute(0, 3, 1, 2), batch, channels, False)

        normalized = self._to_normalized(unfold(point), squeeze)
        if not return_aux:
            return normalized
        # Quantile arrays keep a separate final quantile axis.
        def unfold_knots(value):
            b, o, d, horizon, k = value.shape
            return value.reshape(batch, channels, o, d, horizon, k).permute(0, 4, 2, 3, 1, 5)
        return normalized, unfold(previous_point), unfold(logits), unfold_knots(knots), unfold(past_logits), unfold_knots(past_knots)


class HurdleDataset(TimeSeriesDataset):
    """Completed OD histories at this origin and the same time yesterday."""

    def __init__(self, data, indices, seq_len, horizon, period):
        indices = np.asarray(indices)
        indices = indices[indices - period >= seq_len - 1]
        if period < horizon:
            raise ValueError("Hurdle auxiliary target must be observed: horizon must not exceed period")
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
    for split in splits:
        indices = np.load(folder / f"idx_{split}.npy")
        usable = indices[indices >= config.data.seq_len - 1 + period]
        if split != "train" and len(usable) != len(indices):
            raise ValueError(f"{split} lacks yesterday's history; common evaluation origins are required")
        limit = getattr(config.runtime, f"max_{split}_samples", None)
        ds = HurdleDataset(data, usable[:limit], config.data.seq_len, config.data.horizon, period)
        loaders[f"{split}_loader"] = LoaderAdapter(
            ds, config.training.batch_size, shuffle=split == "train", drop_last=drop,
            num_workers=config.runtime.num_workers, pin_memory=config.runtime.pin_memory,
            logger=logger, name=split,
        )
    return loaders, scaler


class PDRHurdleEngine(BaseEngine_OD):
    def train_batch(self):
        self.model.train()
        for X, label in self._dataloader['train_loader'].get_iterator():
            X, label = self._prepare_batch([X, label])
            self._optimizer.zero_grad(set_to_none=True)
            point, past_point, logits, knots, past_logits, past_knots = self.model(X, return_aux=True)
            previous_label = X['prev_y_od']
            if self._normalize:
                point, label, previous_label = self._inverse_transform([point, label, previous_label], device=self._device)
            mask = self._mask_value.to(point.device)
            main = self.metric.compute_one_batch(point, label, mask, 'train')
            valid = torch.isfinite(label)
            past_valid = torch.isfinite(previous_label)
            occurrence = (F.binary_cross_entropy_with_logits(logits[valid], (label[valid] > .5).float())
                          + F.binary_cross_entropy_with_logits(past_logits[past_valid], (previous_label[past_valid] > .5).float()))
            quantiles = positive_pinball(knots, label, self.model.levels) + positive_pinball(past_knots, previous_label, self.model.levels)
            loss = main + masked_mae(past_point, previous_label, mask) + self.model.occurrence_weight * occurrence + self.model.quantile_weight * quantiles
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite hurdle quantile loss')
            loss.backward()
            if self._clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm, error_if_nonfinite=True)
            self._optimizer.step()
            self._iter_cnt += 1


def setup(config, data_path, adj_path, node_num, device, logger):
    if not config.data.normalize:
        raise ValueError('This candidate uses the original normalized-count protocol')
    folder, data, scaler = _read(data_path, config)
    end = training_end(folder, config.data.horizon)
    active = np.full(data.shape[1:], .5, dtype=np.float64)
    for start in range(0, end, 512):
        active += (data[start:min(start+512, end)] > 0).sum(axis=0)
    ctx = {'activity_prior': active/(end+1.)}
    span = (scaler.data_max_ - scaler.data_min_).numpy()
    if np.any(span <= 0):
        raise ValueError('A nonconstant count channel is required')
    return {**ctx, 'data_min': scaler.data_min_.numpy(), 'data_span': span, 'use_log1p': scaler.use_log1p}


def build_model(config, node_num, **ctx):
    p = config.model.params
    return PDRHurdleQuantile(seq_len=config.data.seq_len, input_dim=node_num, hid_dim=p.hid_dim,
                            node_num=node_num, layer_nums=p.layer_nums, dropout=p.dropout,
                            feature=config.data.input_dim, horizon=config.data.horizon,
                            output_dim=config.data.output_dim, quantile_count=p.quantile_count,
                            event_normalization=p.event_normalization,
                            occurrence_weight=p.occurrence_weight, quantile_weight=p.quantile_weight, **ctx)


def get_recipe():
    return ModelRecipe(build_model=build_model, loss_fn='MAE',
                       metric_list=['MAE', 'MAPE', 'MSE', 'RMSE', 'F1', 'TZR'], od=True, od_cqr=True,
                       load_data=load_data, setup=setup, engine_cls=PDRHurdleEngine)
