from models.layers import ASTGCN_block
import torch
import torch.nn as nn
from models.base import BaseModel
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import calculate_cheb_poly


class ASTGCN(BaseModel):
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
                    self.input_dim,
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
                    self.seq_len // time_stride,
                )
                for _ in range(nb_block - 1)
            ]
        )

        self.final_conv = nn.Conv2d(
            int(self.seq_len / time_stride),
            self.output_dim * self.horizon,
            kernel_size=(1, nb_time_filter),
        )

    def forward(self, x, label=None):  # (b, t, n, f)
        x = x.permute(0, 2, 3, 1)  # (b, n, f, t)

        for block in self.BlockList:
            x = block(x)

        # final_conv maps the time axis to (horizon*output_dim) channels, then we
        # drop the trailing singleton (matching the official squeeze) BEFORE
        # splitting the channel axis, so node and feature axes are never
        # interleaved.  (b, horizon*output_dim, n, 1) -> (b, horizon*output_dim, n)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        output = output.permute(0, 2, 1)  # (b, n, horizon*output_dim)
        output = output.reshape(output.shape[0], self.node_num, self.horizon, self.output_dim)
        return output.permute(0, 2, 1, 3)


def _scaled_combinatorial_laplacian(adj):
    """ASTGCN's official scaled Laplacian: scale the *combinatorial* Laplacian
    ``L = D - W`` by ``2/lambda_max`` and subtract I (guoshnBJTU/ASTGCN-r-pytorch
    ``lib/utils.scaled_Laplacian``).  This differs from the shared ``scalap``
    helper, which scales the *symmetric-normalized* Laplacian."""
    adj = np.asarray(adj, dtype=np.float32)
    adj = np.maximum(adj, adj.T)
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    lambda_max = linalg.eigsh(sp.csr_matrix(L), k=1, which="LM")[0][0]
    return 2.0 * L / lambda_max - np.identity(adj.shape[0], dtype=np.float32)


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    adj = np.zeros((node_num, node_num), dtype=np.float32)
    for n in range(node_num):
        idx = np.nonzero(adj_mx[n])[0]
        adj[n, idx] = 1
    L_tilde = _scaled_combinatorial_laplacian(adj)
    cheb_poly = [
        torch.from_numpy(i).type(torch.FloatTensor).to(device)
        for i in calculate_cheb_poly(L_tilde, config.model.params.order)
    ]
    return {"cheb_poly": cheb_poly}


def build_model(config, node_num, **ctx):
    return ASTGCN(
        node_num=node_num,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        horizon=config.data.horizon,
        device=config.runtime.device,
        cheb_poly=ctx["cheb_poly"],
        order=config.model.params.order,
        nb_block=config.model.params.nb_block,
        nb_chev_filter=config.model.params.nb_chev_filter,
        nb_time_filter=config.model.params.nb_time_filter,
        time_stride=config.model.params.time_stride,
        seq_len=config.data.seq_len,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, setup=setup, init_weights=True)
