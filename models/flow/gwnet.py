from models.layers import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel
from engine.recipe import ModelRecipe
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx


class GWNET(BaseModel):
    """
    Reference code: https://github.com/nnzhan/Graph-WaveNet
    """

    def __init__(
        self,
        supports,
        adp_adj,
        dropout,
        residual_channels,
        dilation_channels,
        skip_channels,
        end_channels,
        horizon=1,
        kernel_size=2,
        blocks=4,
        layers=2,
        **args,
    ):
        super(GWNET, self).__init__(horizon=horizon, **args)
        self.supports = supports
        self.supports_len = len(supports)
        self.adp_adj = adp_adj
        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]
        self.horizon = horizon

        if adp_adj:
            self.nodevec1 = nn.Parameter(torch.randn(self.node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, self.node_num), requires_grad=True)
            self.supports_len += 1

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    GCN(
                        dilation_channels,
                        residual_channels,
                        self.dropout,
                        support_len=self.supports_len,
                    )
                )
        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.output_dim * self.horizon,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, input, label=None):  # (b, t, n, f)
        input = input.transpose(1, 3)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        if self.adp_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        else:
            new_supports = self.supports

        x = self.start_conv(x)

        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # x: (batch, output_dim * horizon, nodes, remaining_time)
        # Take the last time step
        x = x[..., -1]  # (batch, output_dim * horizon, nodes)
        batch_size, _, nodes = x.size()
        x = x.view(batch_size, self.output_dim, self.horizon, nodes)
        x = x.permute(0, 2, 3, 1)  # (batch, horizon, nodes, output_dim)
        return x


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, config.model.params.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    return {"supports": supports}


def build_model(config, node_num, **ctx):
    return GWNET(
        node_num=node_num,
        supports=ctx["supports"],
        adp_adj=config.model.params.adp_adj,
        dropout=config.model.params.dropout,
        residual_channels=config.model.params.init_dim,
        dilation_channels=config.model.params.init_dim,
        skip_channels=config.model.params.skip_dim,
        end_channels=config.model.params.end_dim,
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        horizon=config.data.horizon,
        seq_len=config.data.seq_len,
    )


def get_recipe():
    return ModelRecipe(build_model=build_model, setup=setup)
