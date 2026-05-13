"""FMGCN4 decomposes mobility signals into trend and residual components and models residuals with graph-temporal diffusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


def _row_normalize(adj):
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _graph_matmul_4d(support, x):
    return torch.einsum("nm,btmc->btnc", support, x)


def _moving_average(x, kernel):
    if kernel <= 1:
        return x
    batch_size, seq_len, node_num, feature_dim = x.shape
    reshaped = x.permute(0, 2, 3, 1).reshape(batch_size * node_num * feature_dim, 1, seq_len)
    trend = F.avg_pool1d(F.pad(reshaped, (kernel - 1, 0), mode="replicate"), kernel_size=kernel, stride=1)
    return trend.view(batch_size, node_num, feature_dim, seq_len).permute(0, 3, 1, 2)


class StaticAdaptiveSupports(nn.Module):
    def __init__(self, node_num, graph_dim, support_priors=None):
        super().__init__()
        self.src_node_emb = nn.Parameter(torch.randn(node_num, graph_dim))
        self.dst_node_emb = nn.Parameter(torch.randn(node_num, graph_dim))
        self.prior_gate = nn.Parameter(torch.tensor(0.0))
        if support_priors is None:
            self.register_buffer("support_priors", None)
        else:
            self.register_buffer("support_priors", torch.stack(support_priors, dim=0))

    def forward(self):
        forward = F.softmax(F.relu(self.src_node_emb @ self.dst_node_emb.transpose(0, 1)), dim=-1)
        backward = F.softmax(F.relu(self.dst_node_emb @ self.src_node_emb.transpose(0, 1)), dim=-1)
        if self.support_priors is not None and self.support_priors.shape[0] >= 2:
            mix = torch.sigmoid(self.prior_gate)
            forward = _row_normalize(mix * forward + (1.0 - mix) * self.support_priors[0])
            backward = _row_normalize(mix * backward + (1.0 - mix) * self.support_priors[1])

        supports = []
        if self.support_priors is not None:
            supports.extend(self.support_priors.unbind(0))
        supports.extend([forward, backward])
        return tuple(_row_normalize(s) for s in supports)


class TemporalGLU(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 2, 1)
        x = self.conv(F.pad(x, (self.pad, 0, 0, 0)))
        value, gate = torch.chunk(x, 2, dim=1)
        x = (torch.tanh(value) * torch.sigmoid(gate)).permute(0, 3, 2, 1)
        x = self.dropout(x)
        return self.norm(x + residual)


class DiffusionGraphMix(nn.Module):
    def __init__(self, dim_in, dim_out, order, support_num, dropout):
        super().__init__()
        self.order = order
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear((support_num * order + 1) * dim_in, dim_out)

    def forward(self, x, supports):
        out = [x]
        for support in supports:
            h = x
            for _ in range(self.order):
                h = _graph_matmul_4d(support, h)
                out.append(h)
        out = torch.cat(out, dim=-1)
        return self.dropout(self.proj(out))


class ResidualGraphBlock(nn.Module):
    def __init__(self, hidden_dim, temporal_kernel, temporal_dilation, support_num, diffusion_order, dropout):
        super().__init__()
        self.temporal = TemporalGLU(
            hidden_dim=hidden_dim,
            kernel_size=temporal_kernel,
            dilation=temporal_dilation,
            dropout=dropout,
        )
        self.graph = DiffusionGraphMix(
            dim_in=hidden_dim,
            dim_out=hidden_dim,
            order=diffusion_order,
            support_num=support_num,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, supports):
        h = self.temporal(x)
        g = self.graph(h, supports)
        return self.norm(h + g)


class FMGCN4(BaseModel):
    def __init__(
        self,
        hidden_dim,
        graph_dim,
        num_layers,
        diffusion_order,
        temporal_kernel,
        temporal_dilation,
        trend_kernel,
        trend_window,
        mlp_hidden,
        dropout=0.2,
        support_priors=None,
        **args,
    ):
        super().__init__(**args)
        self.hidden_dim = hidden_dim
        self.trend_kernel = trend_kernel
        self.trend_window = min(trend_window, self.seq_len)
        self.support_builder = StaticAdaptiveSupports(
            node_num=self.node_num,
            graph_dim=graph_dim,
            support_priors=support_priors,
        )
        support_num = (len(support_priors) if support_priors is not None else 0) + 2

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                ResidualGraphBlock(
                    hidden_dim=hidden_dim,
                    temporal_kernel=temporal_kernel,
                    temporal_dilation=temporal_dilation**layer_idx,
                    support_num=support_num,
                    diffusion_order=diffusion_order,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, self.horizon * self.output_dim),
        )
        self.trend_head = nn.Linear(self.trend_window * self.input_dim, self.horizon * self.output_dim)
        self.highway = nn.Linear(self.trend_window * self.input_dim, self.horizon * self.output_dim)
        self.mix_gate = nn.Linear(self.horizon * self.output_dim * 2, self.horizon * self.output_dim)

    def forward(self, source, label=None):
        batch_size, _, node_num, _ = source.shape
        supports = self.support_builder()

        trend = _moving_average(source, self.trend_kernel)
        residual_input = source - trend

        residual_hidden = self.input_proj(residual_input)
        residual_hidden = residual_hidden + self.pos_emb[: self.seq_len].view(1, self.seq_len, 1, self.hidden_dim)
        for block in self.blocks:
            residual_hidden = block(residual_hidden, supports)

        residual_feat = torch.cat(
            [
                residual_hidden[:, -1, :, :],
                residual_hidden.mean(dim=1),
                residual_hidden.max(dim=1).values,
            ],
            dim=-1,
        )
        residual_pred = self.residual_head(residual_feat)

        trend_input = (
            trend[:, -self.trend_window :, :, :]
            .permute(0, 2, 1, 3)
            .reshape(batch_size, node_num, -1)
        )
        trend_pred = self.trend_head(trend_input)

        highway_input = (
            source[:, -self.trend_window :, :, :]
            .permute(0, 2, 1, 3)
            .reshape(batch_size, node_num, -1)
        )
        highway_pred = self.highway(highway_input)

        gate = torch.sigmoid(self.mix_gate(torch.cat([residual_pred, trend_pred], dim=-1)))
        pred = gate * residual_pred + (1.0 - gate) * trend_pred + highway_pred
        pred = pred.view(batch_size, node_num, self.horizon, self.output_dim)
        return pred.permute(0, 2, 1, 3).contiguous()
