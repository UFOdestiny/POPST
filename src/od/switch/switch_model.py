from tkinter import N
import sys
import copy
import os
import math
from turtle import clone
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


class BatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter("weight_self", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x, adj):
        # x: [bs, N, in_features], adj: [N, N]
        # print("BatchGCNConv: ", x.shape)
        input_x = torch.matmul(
            adj, x
        )  # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        # print("input_x", input_x.shape)
        output = self.weight_neigh(
            input_x
        )  # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        # print("output", output.shape)
        if self.weight_self is not None:
            output += self.weight_self(x)  # [bs, N, out_features]
        return output


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.1, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)  # 初始化
        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)  # 初始化
        self.W3 = nn.Parameter(torch.zeros(size=(in_features, in_features)))
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj=None):
        """
        inp: input_fea [B,(N+P+K+M), in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，全部为1.
        """

        batch_size = inp.size()[0]  # [B, N, out_features]
        N = inp.size()[1]  # N 图的节点数

        # zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        Q = torch.matmul(inp, self.W1)
        K = torch.matmul(inp, self.W2)
        V = torch.matmul(inp, self.W3)
        d_model = Q.size()[-1]

        attention = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_model)
        # attention_scores shape: [batch_size*h, seq_len, seq_len]
        attention = F.softmax(
            attention, dim=1
        )  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(
            attention, self.dropout, training=self.training
        )  # dropout，防止过拟合
        h_prime = torch.matmul(attention, V)
        out = F.relu(h_prime) + inp

        return out

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class sgcn_block(nn.Module):
    def __init__(self, in_channel, hidden_channel, bias=False, gcn=False, num_k=4):
        super(sgcn_block, self).__init__()
        self.dropout = 0
        self.gcn = clones(
            BatchGCNConv(in_channel, hidden_channel, bias=bias, gcn=False), num_k
        )

        self.adj_gcn = GraphAttentionLayer(hidden_channel, hidden_channel)

    def forward(self, x, adj):
        # [bs, (N+M+P+K), feature]
        # adj:也是一个列表
        N = x[0].shape[-2]
        batch_size = x[0].shape[0]
        x = [data.reshape(batch_size, N, -1) for data in x]
        x = [model(sdata, s_adj) for model, sdata, s_adj in zip(self.gcn, x, adj)]

        x = torch.cat(x, dim=1)
        x = self.adj_gcn(x, adj)
        len = [q.shape[0] for q in adj]
        x = torch.split(x, len, 1)
        return x


class SWITCH(BaseModel):
    def __init__(self, node_num, adj, args=None):
        super(SWITCH, self).__init__(
            node_num=node_num, input_dim=args.input_dim, output_dim=args.output_dim
        )
        self.dropout = 0.1
        self.adj = [adj]
        # args.gcn["adj"]
        # self.gcn1 = sgcn_block(args.gcn["in_channel"]*args.adjn,args.gcn["hidden_channel"], bias=True, gcn=False,num_k=args.num_trans)
        self.gcn1 = sgcn_block(
            args.gcn["in_channel"],
            args.gcn["hidden_channel"],
            bias=True,
            gcn=False,
            num_k=args.num_trans,
        )

        self.gcn2 = sgcn_block(
            args.gcn["hidden_channel"],
            args.gcn["hidden_channel"],
            bias=True,
            gcn=False,
            num_k=args.num_trans,
        )
        self.gcn3 = sgcn_block(
            args.gcn["in_channel"],
            args.gcn["hidden_channel"],
            bias=True,
            gcn=False,
            num_k=args.num_trans,
        )

        self.gcn4 = sgcn_block(
            args.gcn["hidden_channel"],
            args.gcn["hidden_channel"],
            bias=True,
            gcn=False,
            num_k=args.num_trans,
        )

        self.lstm = clones(
            nn.LSTM(
                args.gcn["hidden_channel"], args.gcn["hidden_channel"], num_layers=2
            ),
            args.num_trans,
        )
        self.lstm2 = clones(
            nn.LSTM(
                args.gcn["hidden_channel"], args.gcn["hidden_channel"], num_layers=2
            ),
            args.num_trans,
        )

        self.tcns = clones(
            nn.Conv1d(
                in_channels=args.tcn["in_channel"],
                out_channels=args.tcn["out_channel"],
                kernel_size=args.tcn["kernel_size"],
                dilation=args.tcn["dilation"],
                padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
            ),
            args.num_trans,
        )

        self.W1s = [
            nn.Parameter(torch.zeros(size=(i, i))).to(args.device)
            for i in args.node_list
        ]
        for parm in self.W1s:
            nn.init.xavier_uniform_(parm, gain=1.414)  # 初始化

        self.W2s = [
            nn.Parameter(torch.zeros(size=(i, i))).to(args.device)
            for i in args.node_list
        ]
        for parm in self.W2s:
            nn.init.xavier_uniform_(parm, gain=1.414)  # 初始化

        self.W12 = [
            nn.Parameter(torch.zeros(size=(i, 64))).to(args.device)
            for i in args.node_list
        ]
        for parm in self.W12:
            nn.init.xavier_uniform_(parm, gain=1.414)

        # self.fcs = nn.ModuleList([nn.Linear(args.gcn["hidden_channel"]*3, 1),nn.Linear(args.gcn["hidden_channel"], 452),nn.Linear(args.gcn["hidden_channel"], 109)])
        self.hidden_channel = args.gcn["hidden_channel"] // 2
        self.args = args

        # self.conv2ds = [nn.Conv2d(in_channels=args.node_list[0], out_channels=1, kernel_size=(1, 1)).to(device="cuda") for i in range(5)]

    def forward(self, x, label=None):
        # X是一个列表，X[B,N,F]
        x = [x]
        N = x[0].shape[2]
        time_length = x[0].shape[1]
        batch = x[0].shape[0]
        node_list = [i.shape[0] for i in self.adj]

        x = [i.view(batch, N, N, 1, time_length) for i in x]

        x_f = [i[:, :, :1, :, :].view(batch, 1, N, time_length) for i in x]
        x_b = [i[:, :, 1:2, :, :].view(batch, 1, N, time_length) for i in x]

        # print("x_f: ",x_f[0].shape)
        # print("x_b: ",x_b[0].shape)

        # for i in range(len(x)):
        #     x[i]=x[i].squeeze(-3)
        #     x[i]=x[i].squeeze(-1)
        #     x[i] = self.conv2ds[i](x[i]).squeeze(2)

        # print("input: ", x[0].shape)
        out = self.gcn1(x_f, self.adj)
        # print("out1 gcn1: ", out[0].shape)

        out = self.gcn2(out, self.adj)  # [B,N,-1]
        # print("out1 gcn2: ", out[0].shape)

        adj_2 = [torch.mm(x_1, x_1.T) for x_1 in self.W12]

        out2 = self.gcn3(x_b, adj_2)
        # print("out2 gcn3: ", out2[0].shape)

        out2 = self.gcn4(out2, adj_2)
        # print("out2 gcn4: ", out2[0].shape)

        # out = [torch.cat([x1,x2],dim=-1) for x1,x2 in zip(out,out2)]
        # print("out: ", out[0].shape)

        out_f = [data.reshape((-1, 1, self.args.gcn["hidden_channel"])) for data in out]
        out_b = [
            data.reshape((-1, 1, self.args.gcn["hidden_channel"])) for data in out2
        ]
        # print("out reshape: ", out_f[0].shape)

        # LSTM dimension?
        out_f = [model(data) for model, data in zip(self.lstm, out_f)]
        out_b = [model(data) for model, data in zip(self.lstm2, out_b)]

        # Tuple
        out_f = [
            data[0].reshape((-1, node_num, self.args.gcn["hidden_channel"]))
            for data, node_num in zip(out_f, node_list)
        ]  # [B,N,1]
        out_b = [
            data[0].reshape((-1, node_num, self.args.gcn["hidden_channel"]))
            for data, node_num in zip(out_b, node_list)
        ]
        # print("out tuple: ", out[0].shape)

        out_f = [torch.matmul(parm, i) for parm, i in zip(self.W1s, out_f)]
        out_b = [torch.matmul(parm, i) for parm, i in zip(self.W2s, out_b)]

        # print("out final: ", out_f[0].shape)
        # print("out final: ", out_b[0].shape)

        # (Batch Size, N, M) * (Batch Size, M, N) -> (Batch Size, N, N)
        out = [torch.bmm(i, j.transpose(1, 2)) for i, j in zip(out_f, out_b)]

        out = [F.relu(i).unsqueeze(1) for i in out]
        # out=[torch.bmm(i, i.transpose(1, 2)) for i in out]

        # print("Result: ", out[0].shape)
        if len(out) == 1:
            return out[0]
        else:
            return out
