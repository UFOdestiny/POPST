"""Shared probabilistic-regression model for the PDR distribution baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.od.pdr import PDR
import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx
from engine.adapters.distributions import PDRRegGaussianEngine
from engine.adapters.distributions import PDRRegLaplaceEngine
from engine.adapters.distributions import PDRRegStudentTEngine


class ODRegimeDistributionHead(nn.Module):
    """Mixture-of-experts head emitting a location and a raw scale."""

    def __init__(self, context_dim, hidden_dim=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.base = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.router = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, context):
        base = self.base(context)
        gate = F.softmax(self.router(context), dim=-1)
        deltas = torch.stack([expert(context) for expert in self.experts], dim=-2)
        return base + (gate.unsqueeze(-1) * deltas).sum(dim=-2)


class PDRRegDistribution(PDR):
    """PDR encoder with a heteroscedastic location-scale regression head.

    Gaussian, Laplace, and Student-t baselines share this exact model.  Their
    only difference is the likelihood used by the engine, making the ablation
    a controlled comparison of predictive distributions.
    """

    cqr_compatible = False

    def __init__(
        self,
        *args,
        context_dim=64,
        num_experts=3,
        head_hidden_dim=128,
        dropout=0.0,
        min_scale=1e-4,
        **kwargs,
    ):
        super().__init__(
            *args,
            context_dim=context_dim,
            num_experts=num_experts,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            **kwargs,
        )
        self.min_scale = float(min_scale)
        if self.min_scale <= 0.0:
            raise ValueError("min_scale must be positive")
        self.head = ODRegimeDistributionHead(
            context_dim=context_dim,
            hidden_dim=head_hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

    def forward(self, X, label=None):
        """Return ``(location, scale)`` in ``(B, horizon, N, N, D)`` layout."""
        X, batch_size, channels, squeeze_back = self._fold_channels(X)
        x = X.permute(0, 2, 3, 1)

        raw = self.head(self._encode(x))
        loc = raw[..., 0].permute(0, 3, 1, 2)
        scale = (F.softplus(raw[..., 1]) + self.min_scale).permute(0, 3, 1, 2)

        loc = self._unfold_channels(loc, batch_size, channels, squeeze_back)
        scale = self._unfold_channels(scale, batch_size, channels, squeeze_back)
        return loc, scale


def make_pdr_reg_distribution(model_cls, engine_cls, *, student_t=False):
    def setup(config, data_path, adj_path, node_num, device, logger):
        adj_mx = load_adj_from_numpy(adj_path)
        adj_mx = adj_mx - np.eye(node_num)
        return {"gso": normalize_adj_mx(adj_mx, "uqgnn")[0]}

    def build_model(config, node_num, **ctx):
        return model_cls(
            A=ctx["gso"],
            node_num=node_num,
            input_dim=config.data.input_dim,
            output_dim=config.data.output_dim,
            seq_len=config.data.seq_len,
            horizon=config.data.horizon,
            context_dim=config.model.params.context_dim,
            zone_embed_dim=config.model.params.zone_embed_dim,
            num_spatial_layers=config.model.params.pdr_num_spatial_layers,
            num_experts=config.model.params.num_experts,
            head_hidden_dim=config.model.params.head_hidden_dim,
            dropout=config.model.params.dropout,
            min_scale=config.model.params.min_scale,
        )

    engine_extras = (
        (lambda config: {"student_df": config.model.params.student_df}) if student_t else None
    )
    return ModelRecipe(
        build_model=build_model,
        engine_cls=engine_cls,
        engine_extras=engine_extras,
        loss_fn="NLL",
        metric_list=["NLL", "MAE", "MAPE", "MSE", "RMSE"],
        od=True,
        od_cqr=True,
        setup=setup,
    )


def get_pdr_reg_gau_recipe():
    return make_pdr_reg_distribution(PDRRegDistribution, PDRRegGaussianEngine)


def get_pdr_reg_lap_recipe():
    return make_pdr_reg_distribution(PDRRegDistribution, PDRRegLaplaceEngine)


def get_pdr_reg_t_recipe():
    return make_pdr_reg_distribution(PDRRegDistribution, PDRRegStudentTEngine, student_t=True)
