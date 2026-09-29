from models.layers import AVWDCRNN
import torch
import torch.nn as nn
from models.base import BaseODModel
from engine.recipe import ModelRecipe


class AGCRN(BaseODModel):
    """
    Reference code: https://github.com/LeiBAI/AGCRN

    Single-channel OD backbone (input_dim = output_dim = node_num) run under the
    channel-as-batch contract of BaseODModel: forward_single receives a 4-D
    (B', T, N, N) tensor (B' = B*D) and returns (B', horizon, N, N).
    """

    def __init__(self, embed_dim, rnn_unit, num_layer, cheb_k, **args):
        super(AGCRN, self).__init__(**args)
        self.node_embed = nn.Parameter(torch.randn(self.node_num, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.input_dim, rnn_unit, cheb_k, embed_dim, num_layer)

        self.end_conv = nn.Conv2d(
            1, self.horizon * self.output_dim, kernel_size=(1, rnn_unit), bias=True
        )

    def forward_single(self, source, label=None):  # (B', t, n, f)
        bs, _, node_num, _ = source.shape
        init_state = self.encoder.init_hidden(bs, node_num)
        output, _ = self.encoder(source, init_state, self.node_embed)
        output = output[:, -1:, :, :]  # (B', 1, N, rnn_unit)
        pred = self.end_conv(output)  # (B', horizon*output_dim, N, 1)
        B, _, N, _ = pred.shape
        pred = pred.view(B, self.horizon, self.output_dim, N)
        pred = pred.permute(0, 1, 3, 2)  # (B', horizon, N, output_dim=N)
        return pred


def build_model(config, node_num, **ctx):
    return AGCRN(
        node_num=node_num,
        input_dim=node_num,
        output_dim=node_num,
        embed_dim=config.model.params.embed_dim,
        rnn_unit=config.model.params.rnn_unit,
        num_layer=config.model.params.num_layer,
        cheb_k=config.model.params.cheb_k,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
    )


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        init_weights=True,
        od=True,
        od_cqr=True,
    )
