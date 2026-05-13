"""FMGCN2 is a multi-scale temporal convolutional graph forecaster designed for robust city-scale mobility."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


def _row_normalize(adj):
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _graph_matmul_2d(support, x):
    return torch.einsum("nm,bcmt->bcnt", support, x)


class AdaptiveSupportBuilder(nn.Module):
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


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.specs = [(2, 1), (3, 1), (2, 2)]
        self.filter_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(1, kernel),
                    dilation=(1, dilation),
                )
                for kernel, dilation in self.specs
            ]
        )
        self.gate_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(1, kernel),
                    dilation=(1, dilation),
                )
                for kernel, dilation in self.specs
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = []
        for (kernel, dilation), filter_conv, gate_conv in zip(
            self.specs, self.filter_convs, self.gate_convs
        ):
            pad = (kernel - 1) * dilation
            x_pad = F.pad(x, (pad, 0, 0, 0))
            branch = torch.tanh(filter_conv(x_pad)) * torch.sigmoid(gate_conv(x_pad))
            outputs.append(branch)
        mixed = sum(outputs) / len(outputs)
        mixed = self.dropout(mixed)
        return mixed + x[:, :, :, -mixed.size(3) :]


class DiffusionConv2d(nn.Module):
    def __init__(self, c_in, c_out, support_num, order, dropout):
        super().__init__()
        self.order = order
        self.proj = nn.Conv2d((support_num * order + 1) * c_in, c_out, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, supports):
        out = [x]
        for support in supports:
            h = x
            for _ in range(self.order):
                h = _graph_matmul_2d(support, h)
                out.append(h)
        out = torch.cat(out, dim=1)
        return self.dropout(self.proj(out))


class GraphTemporalLayer(nn.Module):
    def __init__(self, hidden_dim, skip_dim, support_num, order, dropout):
        super().__init__()
        self.temporal = MultiScaleTemporalBlock(hidden_dim=hidden_dim, dropout=dropout)
        self.graph = DiffusionConv2d(
            c_in=hidden_dim,
            c_out=hidden_dim,
            support_num=support_num,
            order=order,
            dropout=dropout,
        )
        self.skip = nn.Conv2d(hidden_dim, skip_dim, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, supports):
        h = self.temporal(x)
        g = self.graph(h, supports)
        x = self.norm(g + x[:, :, :, -g.size(3) :])
        return x, self.skip(h)


class FMGCN2(BaseModel):
    def __init__(
        self,
        hidden_dim,
        skip_dim,
        end_dim,
        graph_dim,
        num_layers,
        diffusion_order,
        highway_window,
        dropout=0.2,
        support_priors=None,
        **args,
    ):
        super().__init__(**args)
        self.highway_window = min(highway_window, self.seq_len)
        self.support_builder = AdaptiveSupportBuilder(
            node_num=self.node_num,
            graph_dim=graph_dim,
            support_priors=support_priors,
        )
        support_num = (len(support_priors) if support_priors is not None else 0) + 2

        self.input_proj = nn.Conv2d(self.input_dim, hidden_dim, kernel_size=(1, 1))
        self.layers = nn.ModuleList(
            [
                GraphTemporalLayer(
                    hidden_dim=hidden_dim,
                    skip_dim=skip_dim,
                    support_num=support_num,
                    order=diffusion_order,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.end_conv_1 = nn.Conv2d(skip_dim, end_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_dim, self.output_dim * self.horizon, kernel_size=(1, 1))
        self.highway = nn.Linear(
            self.highway_window * self.input_dim,
            self.horizon * self.output_dim,
        )

    def forward(self, source, label=None):
        batch_size, _, node_num, _ = source.shape
        supports = self.support_builder()

        x = source.permute(0, 3, 2, 1)
        x = self.input_proj(x)

        skip = None
        for layer in self.layers:
            x, layer_skip = layer(x, supports)
            skip = layer_skip if skip is None else skip[:, :, :, -layer_skip.size(3) :] + layer_skip

        x = F.gelu(skip)
        x = F.gelu(self.end_conv_1(x))
        x = self.end_conv_2(x)[..., -1]
        pred = x.view(batch_size, self.output_dim, self.horizon, node_num).permute(0, 2, 3, 1)

        highway_input = (
            source[:, -self.highway_window :, :, :]
            .permute(0, 2, 1, 3)
            .reshape(batch_size, node_num, -1)
        )
        pred = pred + self.highway(highway_input).view(batch_size, node_num, self.horizon, self.output_dim).permute(
            0, 2, 1, 3
        )
        return pred.contiguous()
