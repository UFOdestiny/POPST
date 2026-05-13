"""FMGCN1 is a memory-efficient residual graph recurrent forecaster with static-adaptive supports."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


def _row_normalize(adj):
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _graph_matmul(support, x):
    return torch.einsum("nm,bmc->bnc", support, x)


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


class TemporalMixer(nn.Module):
    def __init__(self, hidden_dim, kernel_size, max_dilation, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TemporalGLU(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=max_dilation**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
                h = _graph_matmul(support, h)
                out.append(h)
        out = torch.cat(out, dim=-1)
        return self.dropout(self.proj(out))


class GraphGRUCell(nn.Module):
    def __init__(self, dim_in, hidden_dim, order, support_num, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        gate_dim = dim_in + hidden_dim
        self.gate_conv = DiffusionGraphMix(gate_dim, hidden_dim * 2, order, support_num, dropout)
        self.update_conv = DiffusionGraphMix(gate_dim, hidden_dim, order, support_num, dropout)

    def forward(self, x, state, supports):
        gate_input = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate_conv(gate_input, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, r * state], dim=-1)
        hc = torch.tanh(self.update_conv(candidate, supports))
        return (1.0 - z) * state + z * hc

    def init_hidden_state(self, batch_size, node_num, device):
        return torch.zeros(batch_size, node_num, self.hidden_dim, device=device)


class GraphGRUEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, order, support_num, dropout):
        super().__init__()
        self.cells = nn.ModuleList(
            [
                GraphGRUCell(
                    hidden_dim,
                    hidden_dim,
                    order,
                    support_num=support_num,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, supports):
        batch_size, seq_len, node_num, _ = x.shape
        current = x
        for cell in self.cells:
            outputs = []
            state = cell.init_hidden_state(batch_size, node_num, x.device)
            for step in range(seq_len):
                state = cell(current[:, step, :, :], state, supports)
                outputs.append(state)
            current = torch.stack(outputs, dim=1)
        return current


class TemporalPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class ResidualForecastBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        diffusion_order,
        temporal_kernel,
        temporal_dilation,
        temporal_layers,
        dropout,
        support_num,
    ):
        super().__init__()
        self.temporal_mixer = TemporalMixer(
            hidden_dim=hidden_dim,
            kernel_size=temporal_kernel,
            max_dilation=temporal_dilation,
            num_layers=temporal_layers,
            dropout=dropout,
        )
        self.encoder = GraphGRUEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            order=diffusion_order,
            support_num=support_num,
            dropout=dropout,
        )
        self.temporal_pool = TemporalPool(hidden_dim)
        self.forecast_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backcast_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.res_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, supports):
        mixed = self.temporal_mixer(x)
        encoded = self.encoder(mixed, supports)
        block_summary = self.forecast_proj(
            torch.cat(
                [
                    encoded[:, -1, :, :],
                    self.temporal_pool(encoded),
                    mixed.mean(dim=1),
                ],
                dim=-1,
            )
        )
        backcast = self.backcast_proj(encoded)
        residual = self.res_norm(x - backcast)
        return residual, block_summary


class FMGCN1(BaseModel):
    def __init__(
        self,
        hidden_dim,
        graph_dim,
        num_layers,
        num_blocks,
        diffusion_order,
        temporal_kernel,
        temporal_dilation,
        temporal_layers,
        mlp_hidden,
        highway_window,
        dropout=0.2,
        support_priors=None,
        **args,
    ):
        super().__init__(**args)
        self.hidden_dim = hidden_dim
        self.highway_window = min(highway_window, self.seq_len)
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
                ResidualForecastBlock(
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    diffusion_order=diffusion_order,
                    temporal_kernel=temporal_kernel,
                    temporal_dilation=temporal_dilation,
                    temporal_layers=temporal_layers,
                    dropout=dropout,
                    support_num=support_num,
                )
                for _ in range(num_blocks)
            ]
        )
        decoder_in_dim = hidden_dim * (num_blocks + 2)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, self.horizon * self.output_dim),
        )
        self.highway = nn.Linear(
            self.highway_window * self.input_dim,
            self.horizon * self.output_dim,
        )

    def forward(self, source, label=None):
        batch_size, _, node_num, _ = source.shape
        supports = self.support_builder()

        residual = self.input_proj(source)
        residual = residual + self.pos_emb[: self.seq_len].view(1, self.seq_len, 1, self.hidden_dim)

        block_summaries = []
        for block in self.blocks:
            residual, block_summary = block(residual, supports)
            block_summaries.append(block_summary)

        decoder_input = torch.cat(
            block_summaries
            + [
                residual[:, -1, :, :],
                residual.mean(dim=1),
            ],
            dim=-1,
        )
        pred = self.decoder(decoder_input)

        highway_input = (
            source[:, -self.highway_window :, :, :]
            .permute(0, 2, 1, 3)
            .reshape(batch_size, node_num, -1)
        )
        pred = pred + self.highway(highway_input)
        pred = pred.view(batch_size, node_num, self.horizon, self.output_dim)
        return pred.permute(0, 2, 1, 3).contiguous()
