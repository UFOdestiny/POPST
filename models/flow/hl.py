import torch.nn as nn
from models.base import BaseModel
from engine.recipe import ModelRecipe


class HL(BaseModel):
    def __init__(self, **args):
        super(HL, self).__init__(**args)
        self.L = nn.Linear(self.seq_len, self.horizon)

    def forward(self, input):  # (b, t, n, f)
        x = input.permute(0, 2, 3, 1)
        x = self.L(x)
        x = x.permute(0, 3, 1, 2)
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
    return ModelRecipe(build_model=build_model)
