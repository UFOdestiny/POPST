import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict
from base.model import BaseModel


class DGCRN(BaseModel):
    """
    Reference code: https://github.com/tsinghua-fib-lab/Traffic-Benchmark/tree/master/methods/DGCRN
    """

    # Autoregressive decoder: each step's prediction (width output_dim) is fed
    # back as the next decoder input, and the GRU gate widths assume
    # input_dim == output_dim + time-channels.  Under CQR output_dim widens to
    # 3*F, breaking that invariant (the feedback is not restricted to the
    # median channel).  Reject --cqr rather than mis-feed the decoder.
    cqr_compatible = False

    def __init__(
        self,
        device,
        predefined_adj,
        gcn_depth,
        rnn_size,
        hyperGNN_dim,
        node_dim,
        middle_dim,
        list_weight,
        tpd,
        tanhalpha,
        cl_decay_step,
        dropout,
        **args
    ):
        super(DGCRN, self).__init__(**args)
        self.device = device
        self.predefined_adj = predefined_adj
        self.hidden_size = rnn_size
        self.tpd = tpd
        self.alpha = tanhalpha
        self.cl_decay_step = cl_decay_step
        self.use_curriculum_learning = True

        self.emb1 = nn.Embedding(self.node_num, node_dim)
        self.emb2 = nn.Embedding(self.node_num, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.node_num).to(self.device)

        dims_hyper = [
            self.hidden_size + self.input_dim,
            hyperGNN_dim,
            middle_dim,
            node_dim,
        ]
        self.GCN1_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")
        self.GCN2_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")

        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")
        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")

        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")
        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")

        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")
        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, "hyper")

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        dims = [self.input_dim + self.hidden_size, self.hidden_size]
        self.gz1 = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gz2 = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gr1 = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gr2 = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gc1 = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gc2 = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")

        self.gz1_de = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gz2_de = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gr1_de = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gr2_de = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gc1_de = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")
        self.gc2_de = gcn(dims, gcn_depth, dropout, *list_weight, "RNN")

    def preprocessing(self, adj, predefined_adj):
        adj = adj + torch.eye(self.node_num).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_adj]

    def step(
        self, input, Hidden_State, Cell_State, predefined_adj, type="encoder", i=None
    ):
        x = input
        # ensure x is a 3D tensor (batch, features, nodes)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, nodes) -> (batch, 1, nodes)
        x = x.transpose(1, 2).contiguous()  # (batch, nodes, features)

        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.node_num, self.hidden_size)), 2
        )

        if type == "encoder":
            filter1 = self.GCN1_tg(hyper_input, predefined_adj[0]) + self.GCN1_tg_1(
                hyper_input, predefined_adj[1]
            )

            filter2 = self.GCN2_tg(hyper_input, predefined_adj[0]) + self.GCN2_tg_1(
                hyper_input, predefined_adj[1]
            )

        if type == "decoder":
            filter1 = self.GCN1_tg_de(
                hyper_input, predefined_adj[0]
            ) + self.GCN1_tg_de_1(hyper_input, predefined_adj[1])

            filter2 = self.GCN2_tg_de(
                hyper_input, predefined_adj[0]
            ) + self.GCN2_tg_de_1(hyper_input, predefined_adj[1])

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1)
        )

        adj = F.relu(torch.tanh(self.alpha * a))

        adp = self.preprocessing(adj, predefined_adj[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_adj[1])

        Hidden_State = Hidden_State.view(-1, self.node_num, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.node_num, self.hidden_size)

        combined = torch.cat((x, Hidden_State), -1)

        if type == "encoder":
            z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))

        elif type == "decoder":
            z = torch.sigmoid(self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = torch.sigmoid(self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(1 - z, Cell_State)
        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(
            -1, self.hidden_size
        )

    def compute_future_info(self, his):
        # The official DGCRN conditions its decoder on KNOWN future time-of-day /
        # day-of-week features (channels appended to the input on METR-LA/PEMS).
        # This benchmark's channels are mobility volumes (taxi/fhv/bike) with no
        # calendar features, so there is no future time signal to feed.  Emit
        # zeros (no time conditioning) rather than reconstructing time-of-day
        # from mobility magnitudes, which would inject target-correlated noise
        # into the decoder at every step.  The 2 zero channels keep the decoder
        # input width consistent (output_dim + 2 == input_dim).
        b, f, n, t = his.shape
        return torch.zeros((b, 2, n, self.seq_len), device=his.device)

    def forward(
        self, input, label=None, batches_seen=None, task_level=12
    ):  # (b, t, n, f)
        x = input.transpose(1, 3)
        label = label.transpose(1, 3)

        batch_size = x.size(0)
        Hidden_State, Cell_State = self.initHidden(
            batch_size * self.node_num, self.hidden_size
        )

        outputs = None
        for i in range(self.seq_len):
            # print(i,x.shape)
            Hidden_State, Cell_State = self.step(
                x[..., i],
                Hidden_State,
                Cell_State,
                self.predefined_adj,
                "encoder",
                i,
            )
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        timeofday = self.compute_future_info(x[:, :, :, :])
        decoder_input = torch.zeros(
            (batch_size, self.output_dim, self.node_num), device=self.device
        )
        outputs_final = []

        for i in range(task_level):
            # print(i)
            try:
                decoder_input = torch.cat([decoder_input, timeofday[..., i]], dim=1)
            except:
                print(decoder_input.shape, timeofday.shape)
                sys.exit(0)

            Hidden_State, Cell_State = self.step(
                decoder_input,
                Hidden_State,
                Cell_State,
                self.predefined_adj,
                "decoder",
                None,
            )

            decoder_output = self.fc_final(Hidden_State)
            decoder_input = decoder_output.view(
                batch_size, self.node_num, self.output_dim
            ).transpose(1, 2)
            outputs_final.append(decoder_output)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    decoder_input = label[:, :, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)
        # print(outputs_final.shape,task_level)
        outputs_final = outputs_final.view(
            batch_size, self.node_num, task_level, self.output_dim
        ).transpose(1, 2)
        return outputs_final

    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device)
            )
            Cell_State = Variable(torch.zeros(batch_size, hidden_size).to(self.device))
            nn.init.orthogonal_(Hidden_State)
            nn.init.orthogonal_(Cell_State)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_step / (
            self.cl_decay_step + np.exp(batches_seen / self.cl_decay_step)
        )


class gcn(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(gcn, self).__init__()
        if type == "RNN":
            self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == "hyper":
            self.gconv = gconv_hyper()
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("fc1", nn.Linear((gdep + 1) * dims[0], dims[1])),
                        ("sigmoid1", nn.Sigmoid()),
                        ("fc2", nn.Linear(dims[1], dims[2])),
                        ("sigmoid2", nn.Sigmoid()),
                        ("fc3", nn.Linear(dims[2], dims[3])),
                    ]
                )
            )
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x, adj):
        # if x.shape[2] > x.shape[1]:
        #     x = x[..., : x.shape[1]]
        h = x
        out = [h]
        if self.type_GNN == "RNN":
            for _ in range(self.gdep):
                h = (
                    self.alpha * x
                    + self.beta * self.gconv(h, adj[0])
                    + self.gamma * self.gconv_preA(h, adj[1])
                )
                out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)

        ho = torch.cat(out, dim=-1)

        if self.type_GNN == "hyper":
            ho = ho[..., : self.mlp[0].in_features]
        else:
            ho = ho[..., : self.mlp.in_features]
        ho = self.mlp(ho)
        return ho


class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("nvc,nvw->nwc", (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("nvc,vw->nwc", (x, A))
        return x.contiguous()
