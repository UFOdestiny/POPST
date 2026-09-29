"""Shared experiment wiring for PDR single-component ablations."""

import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx
from models.od.pdr import PDR
from engine.adapters.pdr import PDR_Engine


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    return dict(gso=gso, device=device)


def make_pdr_ablation(model_cls, engine_cls):
    """Run one PDR ablation with the baseline's data and engine settings."""

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
        )

    return ModelRecipe(
        build_model=build_model,
        loss_fn="NLL",
        metric_list=["NLL", "MAE", "MAPE", "MSE", "RMSE"],
        od=True,
        od_cqr=True,
        engine_cls=engine_cls,
        setup=setup,
    )


class PDR_no_context(PDR):
    """Ablate origin, destination, and global aggregate temporal contexts."""

    def __init__(self, *args, **kwargs):
        kwargs["use_aggregate_context"] = False
        super().__init__(*args, **kwargs)


def get_pdr_no_context_recipe():
    return make_pdr_ablation(PDR_no_context, PDR_Engine)


class PDR_no_moe(PDR):
    """Ablate the regime router and all expert residual branches."""

    def __init__(self, *args, **kwargs):
        kwargs["use_moe"] = False
        super().__init__(*args, **kwargs)


def get_pdr_no_moe_recipe():
    return make_pdr_ablation(PDR_no_moe, PDR_Engine)


class PDR_no_spatial(PDR):
    """Ablate PDR's graph diffusion blocks and their adjacency inputs."""

    def __init__(self, *args, **kwargs):
        kwargs["use_spatial_mixing"] = False
        super().__init__(*args, **kwargs)


def get_pdr_no_spatial_recipe():
    return make_pdr_ablation(PDR_no_spatial, PDR_Engine)


class PDR_no_zone_embed(PDR):
    """Ablate learned origin/destination identities, retaining demand context."""

    def __init__(self, *args, **kwargs):
        kwargs["use_zone_embeddings"] = False
        super().__init__(*args, **kwargs)


def get_pdr_no_zone_embed_recipe():
    return make_pdr_ablation(PDR_no_zone_embed, PDR_Engine)
