from models.layers import ASTGCN_block
import torch
import torch.nn as nn
from models.base import BaseODModel
import numpy as np
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx, calculate_cheb_poly


class ASTGCN(BaseODModel):
    """
    Reference code: https://github.com/guoshnBJTU/ASTGCN-r-pytorch
    """

    def __init__(
        self,
        device,
        cheb_poly,
        order,
        nb_block,
        nb_chev_filter,
        nb_time_filter,
        time_stride,
        **args,
    ):
        super(ASTGCN, self).__init__(**args)

        self.BlockList = nn.ModuleList(
            [
                ASTGCN_block(
                    device,
                    self.node_num,
                    order,
                    nb_chev_filter,
                    nb_time_filter,
                    time_stride,
                    cheb_poly,
                    self.node_num,
                    self.seq_len,
                )
            ]
        )
        self.BlockList.extend(
            [
                ASTGCN_block(
                    device,
                    nb_time_filter,
                    order,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    cheb_poly,
                    self.node_num,
                    (self.seq_len + time_stride - 1) // time_stride,
                )
                for _ in range(nb_block - 1)
            ]
        )

        # The ST blocks collapse the feature (destination) axis into
        # nb_time_filter; the (1, nb_time_filter) kernel reduces that to 1, so
        # final_conv must re-emit both the forecast horizon and the N
        # destinations.  Output horizon*output_dim channels, then split.
        self.final_conv = nn.Conv2d(
            (self.seq_len + time_stride - 1) // time_stride,
            self.horizon * self.output_dim,
            kernel_size=(1, nb_time_filter),
        )

    def forward_single(self, x, label=None):  # (B', t, n, f)
        b = x.shape[0]
        x = x.permute(0, 2, 3, 1)  # (B', n, f, t)

        for block in self.BlockList:
            x = block(x)

        # x: (B', n, nb_time_filter, t)
        output = self.final_conv(x.permute(0, 3, 1, 2))  # (B', horizon*n, n, 1)
        output = output.squeeze(-1)  # (B', horizon*output_dim, n)
        output = output.reshape(b, self.horizon, self.output_dim, self.node_num)
        return output.permute(0, 1, 3, 2)


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    adj = np.zeros((node_num, node_num), dtype=np.float32)
    for n in range(node_num):
        idx = np.nonzero(adj_mx[n])[0]
        adj[n, idx] = 1
    L_tilde = normalize_adj_mx(adj, "scalap")[0]
    cheb_poly = [
        torch.from_numpy(i).type(torch.FloatTensor).to(device)
        for i in calculate_cheb_poly(L_tilde, config.model.params.order)
    ]
    return dict(cheb_poly=cheb_poly)


def build_model(config, node_num, **ctx):
    return ASTGCN(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
        device=config.runtime.device,
        cheb_poly=ctx["cheb_poly"],
        order=config.model.params.order,
        nb_block=config.model.params.nb_block,
        nb_chev_filter=config.model.params.nb_chev_filter,
        nb_time_filter=config.model.params.nb_time_filter,
        time_stride=config.model.params.time_stride,
    )


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="MAE",
        od=True,
        od_cqr=True,
        init_weights=True,
        setup=setup,
    )
