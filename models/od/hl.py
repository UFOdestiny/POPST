import torch.nn as nn
from models.base import BaseODModel
from engine.recipe import ModelRecipe


class HL(BaseODModel):
    """Historical Linear: a learnable linear map over the input window, applied
    per origin-destination pair.  Channel-as-batch (see BaseODModel)."""

    def __init__(self, **args):
        super(HL, self).__init__(**args)
        self.L = nn.Linear(self.seq_len, self.horizon)

    def forward_single(self, x, label=None):  # (B', T, N, N)
        x = x.permute(0, 2, 3, 1)  # (B', N, N, T)
        x = self.L(x)  # (B', N, N, H)
        x = x.permute(0, 3, 1, 2)  # (B', H, N, N)
        return x


def build_model(config, node_num, **ctx):
    return HL(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, loss_fn="MAE", od=True, od_cqr=True)
