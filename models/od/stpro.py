"""STPro's released prediction path adapted to the project's OD windows.

Source: https://github.com/AIMS-SDU/STPro (main, retrieved 2026-09-29).
The release uses learned prototype projections and O-query/D-key/value cross
attention. It does not activate its commented clustering alternatives. See
README.md for the dimension, attention-layout, and output-padding repairs.

Copyright (c) 2026 SDU AIMS Lab. Adapted under the MIT license; the complete
upstream notice is retained in licenses/STPro.txt.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from engine.recipe import ModelRecipe
from models.base import BaseODModel


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.fc1(x)))


class PrototypeProjection(nn.Module):
    """The release's shared MLP and separate O/D two-stage projections."""

    def __init__(self, node_num, prototype_num, dropout=0.2):
        super().__init__()
        self.MLP = MultiLayerPerceptron(node_num, node_num, dropout)
        rank = max(prototype_num - 1, 1)
        self.W = nn.Parameter(torch.randn(node_num, rank))
        self.Wdo = nn.Parameter(torch.randn(node_num, rank))
        self.WC = nn.Parameter(torch.randn(rank, prototype_num))
        self.WCdo = nn.Parameter(torch.randn(rank, prototype_num))

    def forward(self, x, x_do):
        # (B,T,N,N) -> (B,T,K,N), matching the upstream einops rearrange.
        origin = F.relu(self.MLP(x) @ self.W @ self.WC).transpose(-2, -1)
        destination = F.relu(self.MLP(x_do) @ self.Wdo @ self.WCdo).transpose(-2, -1)
        return origin, destination


class DualInfoTransformer(nn.Module):
    """Released O/D Conv1d attention with explicit node and head axes."""

    def __init__(self, hidden_dim=128, num_heads=2):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("STPro hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        projection = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.PReLU(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.PReLU(hidden_dim),
        )
        self.od_linears = nn.ModuleList([copy.deepcopy(projection) for _ in range(3)])
        self.do_linears = nn.ModuleList([copy.deepcopy(projection) for _ in range(3)])
        self.od_conv = copy.deepcopy(projection)
        self.do_conv = copy.deepcopy(projection)

    def _project(self, linears, x):
        batch, nodes, _ = x.shape
        x = x.transpose(1, 2)
        return [
            layer(x).reshape(batch, self.num_heads, self.head_dim, nodes).transpose(-2, -1)
            for layer in linears
        ]

    def _attend(self, query, key, value, output_layer):
        batch, _, nodes, _ = query.shape
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        attended = scores.softmax(dim=-1) @ value
        attended = attended.transpose(-2, -1).reshape(batch, self.hidden_dim, nodes)
        return output_layer(attended).transpose(1, 2)

    def forward(self, origin, destination):
        oq, ok, ov = self._project(self.od_linears, origin)
        dq, dk, dv = self._project(self.do_linears, destination)
        return (
            self._attend(oq, dk, dv, self.od_conv),
            self._attend(dq, ok, ov, self.do_conv),
        )


class Dual(nn.Module):
    def __init__(self, seq_len, node_num, hidden_dim, num_heads):
        super().__init__()
        self.W = nn.Parameter(torch.randn(seq_len * node_num, hidden_dim))
        self.interact = DualInfoTransformer(hidden_dim, num_heads)

    def forward(self, origin, destination):
        batch, _, prototypes, _ = origin.shape
        origin = origin.transpose(1, 2).reshape(batch, prototypes, -1)
        destination = destination.transpose(1, 2).reshape(batch, prototypes, -1)
        origin = F.relu(origin @ self.W)
        destination = F.relu(destination @ self.W)
        # Upstream STProm predicts from the first attention result only.
        attended, _ = self.interact(origin, destination)
        return attended.unsqueeze(1)


class STPro(BaseODModel):
    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon,
                 hidden_dim=128, num_heads=2, prototype_num=0, dropout=0.2):
        super().__init__(node_num, input_dim, output_dim, seq_len, horizon)
        if node_num <= 0 or seq_len <= 0 or horizon <= 0 or not 0 <= dropout < 1:
            raise ValueError("STPro needs positive dimensions and 0 <= dropout < 1")
        if prototype_num < 0 or prototype_num > node_num:
            raise ValueError("STPro prototype_num must be between 0 and node_num")
        self.prototype_num = prototype_num or math.ceil(node_num / 3)
        self.expansion = math.ceil(node_num / self.prototype_num)
        self.hyperg = PrototypeProjection(node_num, self.prototype_num, dropout)
        self.encoder = Dual(seq_len, node_num, hidden_dim, num_heads)
        # The original 69-node, 23-prototype case has expansion=3. Padding
        # the output allows other node counts without merging any OD cells.
        self.end_conv = nn.Conv2d(
            1, horizon * node_num * self.expansion, kernel_size=(1, hidden_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Random positive biases accumulate through the projection/attention
        # stack and can overflow the log1p scaler's count-space inverse.
        # Keep Xavier matrices, zero affine biases and the standard PReLU slope.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        for module in self.modules():
            if isinstance(module, nn.PReLU):
                nn.init.constant_(module.weight, 0.25)

    def forward_single(self, x, label=None):
        if x.shape[1:] != (self.seq_len, self.node_num, self.node_num):
            raise ValueError("STPro input must match (B,seq_len,node_num,node_num)")
        origin, destination = self.hyperg(x, x.transpose(-2, -1))
        hidden = self.encoder(origin, destination)
        output = self.end_conv(hidden).squeeze(-1)
        output = output.reshape(
            x.shape[0], self.horizon, self.node_num,
            self.prototype_num * self.expansion,
        )
        return output[..., :self.node_num].transpose(-2, -1)


def build_model(config, node_num, **ctx):
    return STPro(
        node_num=node_num,
        input_dim=node_num,
        output_dim=config.data.output_dim,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
        **config.model.params.to_dict(),
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, od=True, od_cqr=True)
