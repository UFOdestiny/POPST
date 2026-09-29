"""OD-CED using only OD history, adapted from the author's public code.

Source: https://github.com/luckyyangrun/OD-CED (main, retrieved 2026-09-29).
The released MSE/STAR-embedding/decoder/Conv2d path is preserved. Semantic
coarsening is fitted only on training OD counts; no geographic, POI, or clock
inputs are used. This is an OD-only adaptation, not the full paper method.
"""

import math

import numpy as np
import torch
from torch import nn
from einops import rearrange

from data.loader import _read
from data.temporal import training_end
from engine.recipe import ModelRecipe
from engine.trainer import BaseEngine_OD
from models.base import BaseODModel


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None):
        super(CAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v, trans_mat):
        q_B, q_N, q_C = q.shape
        k_B, k_N, k_C = k.shape
        v_B, v_N, v_C = v.shape

        q = self.q(q).reshape(q_B, q_N, self.num_heads, q_C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(k_B, k_N, self.num_heads, k_C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(v_B, v_N, self.num_heads, v_C // self.num_heads).permute(0, 2, 1, 3)
        trans_mat = trans_mat.unsqueeze(1).repeat(1, self.num_heads, 1, 1)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~trans_mat.bool(), float("-inf"))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(q_B, q_N, v_C)
        x = self.proj(x)

        return x


class DECODE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1, qkv_bias=False, qk_scale=None, drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(DECODE, self).__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.c_norm1 = norm_layer(dim)
        self.c_attn = CAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.c_norm2 = norm_layer(dim)
        self.c_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, dec_inputs, enc_outputs, trans_mat):

        x = self.c_attn(q=self.c_norm1(dec_inputs),
                    k=self.c_norm1(enc_outputs),
                    v=self.c_norm1(enc_outputs), trans_mat = trans_mat) + dec_inputs
        x = self.c_mlp(self.c_norm2(x))

        return x


class STAR_Embed(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(STAR_Embed, self).__init__()
        self.embed_o = nn.Linear(seq_len, embed_dim)
        self.embed_d = nn.Linear(seq_len, embed_dim)

    def forward(self, data):

        tmp_o = self.embed_o(rearrange(data, 'b t o d -> b d o t'))
        tmp_d = self.embed_d(rearrange(data, 'b t o d -> b o d t'))
        star_patch = torch.cat([tmp_o, tmp_d], dim=2)
        x= torch.sum(star_patch, dim=2)
        return x


class PHEAD(nn.Module):
    def __init__(self, embed_dim, num_node):
        super(PHEAD, self).__init__()
        self.mlp = Mlp(in_features = embed_dim, out_features = num_node, hidden_features=embed_dim)
    def forward(self, x):
        out = self.mlp(x)
        out =  out.unsqueeze(1)
        return out


def semantic_coarsening(mean_counts, dense_quantile=0.85, max_iter=100, tolerance=1e-6):
    """Seed dense cells and propagate labels using training OD flows alone.

    The author's repository expects an external new_ids.npy but does not
    publish the preprocessing script. This adapter implements its required
    mapping with the paper's labeling/propagation structure and a semantic
    transition only. Every dense cell remains its own community seed.
    """
    if not 0 <= dense_quantile < 1 or max_iter <= 0 or tolerance <= 0:
        raise ValueError("Invalid OD-CED coarsening settings")
    mean_counts = np.asarray(mean_counts, dtype=np.float64)
    if mean_counts.ndim != 2 or mean_counts.shape[0] != mean_counts.shape[1]:
        raise ValueError("OD-CED needs a square mean OD matrix")
    if not np.isfinite(mean_counts).all() or np.any(mean_counts < 0):
        raise ValueError("OD-CED coarsening requires finite nonnegative counts")
    nodes = len(mean_counts)
    volume = mean_counts.sum(0) + mean_counts.sum(1)
    dense_count = max(1, math.ceil(nodes * (1 - dense_quantile)))
    dense = np.argsort(-volume, kind="stable")[:dense_count]
    sparse = np.ones(nodes, dtype=bool)
    sparse[dense] = False
    weights = mean_counts + mean_counts.T
    np.fill_diagonal(weights, 0)
    total = weights.sum(1, keepdims=True)
    transition = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    labels = np.zeros((nodes, dense_count), dtype=np.float64)
    labels[dense] = np.eye(dense_count)
    for _ in range(max_iter):
        updated = transition @ labels
        updated[dense] = np.eye(dense_count)
        if np.max(np.abs(updated - labels)) < tolerance:
            labels = updated
            break
        labels = updated
    # A disconnected zero-flow cell has no identifiable community; stable
    # argmax assigns it to the highest-volume seed without external data.
    groups = labels.argmax(1)
    assignment = np.eye(dense_count, dtype=np.float32)[groups]
    fine_mask = np.outer(~sparse, ~sparse)
    coarse_mask = np.zeros((dense_count, dense_count), dtype=bool)
    coarse_mask[groups[sparse], groups[sparse]] = True
    if not coarse_mask.any():
        coarse_mask[:] = True
    return assignment, fine_mask, coarse_mask


def setup(config, data_path, adj_path, node_num, device, logger):
    folder, data, scaler = _read(data_path, config)
    end = training_end(folder, config.data.horizon)
    summed = np.zeros((node_num, node_num), dtype=np.float64)
    for start in range(0, end, 256):
        chunk = np.asarray(data[start:min(start + 256, end)])
        counts = scaler.inverse_transform(chunk, device="cpu") if config.data.normalize else chunk
        summed += counts.sum(axis=(0, 3), dtype=np.float64)
    params = config.model.params
    assignment, fine_mask, coarse_mask = semantic_coarsening(
        summed / end, params.dense_quantile, params.coarsening_max_iter,
        params.coarsening_tolerance,
    )
    if logger:
        logger.info(
            f"OD-CED semantic coarsening: {node_num} cells -> {assignment.shape[1]} groups; "
            f"fitted on training observations [0,{end})"
        )
    return dict(assignment=assignment, fine_mask=fine_mask, coarse_mask=coarse_mask, scaler=scaler)


class ODCED(BaseODModel):
    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon,
                 assignment, fine_mask, coarse_mask, scaler, normalize=True,
                 embed_dim=64, num_heads=4):
        super().__init__(node_num, input_dim, output_dim, seq_len, horizon)
        if embed_dim < 2 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("OD-CED embed_dim must be divisible by num_heads")
        assignment = torch.as_tensor(assignment, dtype=torch.float32)
        if assignment.ndim != 2 or assignment.shape[0] != node_num:
            raise ValueError("OD-CED assignment must have one row per OD cell")
        if not torch.all(assignment.sum(1) == 1) or not torch.all((assignment == 0) | (assignment == 1)):
            raise ValueError("OD-CED assignment must be one-hot")
        coarse_nodes = assignment.shape[1]
        self.register_buffer("assignment", assignment)
        self.register_buffer("fine_mask", torch.as_tensor(fine_mask, dtype=torch.bool))
        self.register_buffer("coarse_mask", torch.as_tensor(coarse_mask, dtype=torch.bool))
        self.register_buffer("norm_min", scaler.data_min_.clone())
        span = scaler.data_max_ - scaler.data_min_
        self.register_buffer("norm_span", torch.where(span == 0, torch.ones_like(span), span))
        self.register_buffer("norm_log1p", torch.tensor(scaler.use_log1p))
        self.normalize = normalize
        self.enc_star_embed = STAR_Embed(seq_len, embed_dim)
        self.decode = DECODE(embed_dim, num_heads=num_heads)
        self.out_layer_dec = PHEAD(embed_dim, node_num)
        self.out_layer_enc = PHEAD(embed_dim, coarse_nodes)
        self.outconv = self._output_conv(seq_len, horizon)
        self.outconv_enc = self._output_conv(seq_len, horizon)
        self.dec_embedding = nn.Parameter(torch.randn(node_num, embed_dim))
        # No absolute clock inputs: forecast-step embeddings are model weights.
        self.horizon_embedding = nn.Parameter(torch.randn(horizon, embed_dim))

    @staticmethod
    def _output_conv(seq_len, horizon):
        return nn.Sequential(
            nn.Conv2d(seq_len + 1, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 16, 1), nn.Conv2d(16, horizon, 1),
        )

    def inverse(self, x):
        if not self.normalize:
            return x
        minimum, span = self.norm_min, self.norm_span
        if minimum.ndim and x.shape[-1] == 1:
            minimum, span = minimum[:1], span[:1]
        x = x * span + minimum
        return torch.expm1(x) if self.norm_log1p else x

    def transform(self, x):
        if not self.normalize:
            return x
        if self.norm_log1p:
            x = torch.log1p(x)
        minimum, span = self.norm_min, self.norm_span
        if minimum.ndim and x.shape[-1] == 1:
            minimum, span = minimum[:1], span[:1]
        return (x - minimum) / span

    def aggregate(self, x):
        # Sum original counts before applying the project scaler: adding
        # log-normalized OD cells would change the coarsened demand totals.
        counts = self.inverse(x)
        coarse = torch.einsum("nm,btnkc,kl->btmlc", self.assignment, counts, self.assignment)
        return self.transform(coarse)

    def forward(self, x, label=None, return_aux=False):
        squeeze_back = x.ndim == 4
        if squeeze_back:
            x = x.unsqueeze(-1)
        coarse = self.aggregate(x)
        fine, batch, channels, _ = self._fold_channels(x)
        coarse, _, _, _ = self._fold_channels(coarse)
        encoded = self.enc_star_embed(coarse)
        trans_mat = self.assignment.unsqueeze(0).expand(len(fine), -1, -1)
        # Evaluate each horizon query independently while retaining the
        # released current-history-plus-prediction Conv2d head.
        predictions = []
        for step in range(self.horizon):
            query = self.dec_embedding + self.horizon_embedding[step]
            decoded = self.decode(query.unsqueeze(0).expand(len(fine), -1, -1), encoded, trans_mat)
            preliminary = self.out_layer_dec(decoded)
            predictions.append(self.outconv(torch.cat((fine, preliminary), dim=1))[:, step:step + 1])
        point = self._unfold_channels(torch.cat(predictions, dim=1), batch, channels, squeeze_back)
        if not return_aux:
            return point
        coarse_prediction = self.outconv_enc(torch.cat((coarse, self.out_layer_enc(encoded)), dim=1))
        return point, self._unfold_channels(coarse_prediction, batch, channels, squeeze_back)


def masked_mse(prediction, target, mask):
    mask = mask.expand_as(prediction) & torch.isfinite(target)
    if not mask.any():
        # Empty source supervision subsets have zero weight, not NaN loss.
        return prediction.sum() * 0
    return (prediction[mask] - target[mask]).square().mean()


class ODCED_Engine(BaseEngine_OD):
    def train_batch(self):
        self.model.train()
        for x, label in self._dataloader["train_loader"].get_iterator():
            x, label = self._prepare_batch([x, label])
            self._optimizer.zero_grad(set_to_none=True)
            point, coarse = self.model(x, return_aux=True)
            coarse_label = self.model.aggregate(label)
            point, coarse, label, coarse_label = [
                self.model.inverse(value) for value in (point, coarse, label, coarse_label)
            ]
            self.metric.compute_one_batch(point, label, self._mask_value.to(point.device), "train")
            fine_mask = self.model.fine_mask[None, None, :, :, None]
            coarse_mask = self.model.coarse_mask[None, None, :, :, None]
            loss = masked_mse(point, label, fine_mask) + masked_mse(coarse, coarse_label, coarse_mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite OD-CED masked MSE")
            loss.backward()
            if self._clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm, error_if_nonfinite=True)
            self._optimizer.step()
            self._iter_cnt += 1


def build_model(config, node_num, **ctx):
    return ODCED(
        node_num=node_num, input_dim=node_num, output_dim=config.data.output_dim,
        seq_len=config.data.seq_len, horizon=config.data.horizon,
        normalize=config.data.normalize, embed_dim=config.model.params.embed_dim,
        num_heads=config.model.params.num_heads, **ctx,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, setup=setup, engine_cls=ODCED_Engine,
                       od=True, od_cqr=True, init_weights=True)
