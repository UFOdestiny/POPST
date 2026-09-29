"""Point-regression variant of PDR."""

import torch
import torch.nn as nn
from models.od.pdr import PDR
import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx
from engine.calibration.zero_cqr import ZeroCQREngine


class ODRegimeRegressionHead(nn.Module):
    """PDR's mixture-of-experts head with one point output per OD pair."""

    def __init__(self, context_dim, hidden_dim=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.base = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, context):
        base = self.base(context)
        gate = nn.functional.softmax(self.router(context), dim=-1)
        deltas = torch.stack([expert(context) for expert in self.experts], dim=-2)
        return base + (gate.unsqueeze(-1) * deltas).sum(dim=-2)


class PDRReg(PDR):
    """PDR encoder trained as an ordinary point-regression OD model.

    The probabilistic PDR head emits three ZINB parameters.  This variant keeps
    the same encoder, router, and expert layout, but each base/expert branch
    emits one unconstrained regression value.  Consequently ``forward``
    returns the standard OD tensor expected by ``BaseEngine_OD`` and can be
    trained with ordinary losses such as MAE or MSE.
    """

    def __init__(
        self,
        *args,
        context_dim=64,
        num_experts=3,
        head_hidden_dim=128,
        dropout=0.0,
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
        self.head = ODRegimeRegressionHead(
            context_dim=context_dim,
            hidden_dim=head_hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

    def forward(self, X, label=None):
        """Return point forecasts shaped ``(B, horizon, N, N, D)``."""
        X, batch_size, channels, squeeze_back = self._fold_channels(X)
        x = X.permute(0, 2, 3, 1)

        context = self._encode(x)
        pred = self.head(context)[..., 0].permute(0, 3, 1, 2)
        return self._unfold_channels(pred, batch_size, channels, squeeze_back)


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    return {"gso": gso}


def build_model(config, node_num, **ctx):
    return PDRReg(
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
    )


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        metric_list=["MAE", "MAPE", "MSE", "RMSE"],
        od=True,
        od_cqr=True,
        setup=setup,
    )


def get_post_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        metric_list=ZeroCQREngine.DEFAULT_METRICS,
        od=True,
        od_cqr=True,
        engine_cls=ZeroCQREngine,
        engine_extras=lambda config: {
            "zero_cqr_alpha": config.calibration.options.zero_cqr_alpha,
            "zero_cqr_gate_quantile": config.calibration.options.zero_cqr_gate_quantile,
            "zero_cqr_grid_size": config.calibration.options.zero_cqr_grid_size,
            "zero_cqr_min_group": config.calibration.options.zero_cqr_min_group,
            "zero_cqr_mse_weight": config.calibration.options.zero_cqr_mse_weight,
            "zero_cqr_aux_epochs": config.calibration.options.zero_cqr_aux_epochs,
            "zero_cqr_aux_samples": config.calibration.options.zero_cqr_aux_samples,
            "zero_cqr_period": config.calibration.options.zero_cqr_period,
            "zero_cqr_enable_online": not config.calibration.options.zero_cqr_disable_online,
            "zero_cqr_zero_floor": config.calibration.options.zero_cqr_zero_floor,
            "zero_cqr_active_bins": config.calibration.options.zero_cqr_active_bins,
        },
        setup=setup,
    )
