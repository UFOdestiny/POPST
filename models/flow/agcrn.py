from models.layers import AVWDCRNN
import torch
import torch.nn as nn
from models.base import BaseModel
from engine.recipe import ModelRecipe


class AGCRN(BaseModel):
    """
    Reference code: https://github.com/LeiBAI/AGCRN
    """

    def __init__(self, embed_dim, rnn_unit, num_layer, cheb_k, **args):
        super(AGCRN, self).__init__(**args)
        self.node_embed = nn.Parameter(torch.randn(self.node_num, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.input_dim, rnn_unit, cheb_k, embed_dim, num_layer)

        self.end_conv = nn.Conv2d(
            1, self.horizon * self.output_dim, kernel_size=(1, rnn_unit), bias=True
        )

    def forward(self, source, label=None):  # (b, t, n, f)
        bs, _, node_num, _ = source.shape
        init_state = self.encoder.init_hidden(bs, node_num)
        output, _ = self.encoder(source, init_state, self.node_embed)
        output = output[:, -1:, :, :]  # (B, 1, N, rnn_unit)
        pred = self.end_conv(output)  # (B, horizon*out_dim, N, 1)
        # Conv channels are laid out (horizon, output_dim); split that axis BEFORE
        # moving N into place so the horizon/feature axes are not interleaved
        # (matches the official reshape: (B, T*C, N, 1) -> (B, T, C, N) -> (B, T, N, C)).
        B = pred.shape[0]
        pred = pred.squeeze(-1).reshape(B, self.horizon, self.output_dim, self.node_num)
        pred = pred.permute(0, 1, 3, 2)  # (B, horizon, N, output_dim)

        return pred


def build_model(config, node_num, **ctx):
    return AGCRN(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        embed_dim=config.model.params.embed_dim,
        rnn_unit=config.model.params.rnn_unit,
        num_layer=config.model.params.num_layer,
        cheb_k=config.model.params.cheb_k,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, init_weights=True)
