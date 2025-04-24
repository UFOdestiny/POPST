import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

import numpy as np
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d, BatchNorm1d

from base.model import BaseModel


class PGNN(BaseModel):
    def __init__(self, device, num_nodes, cluster_nodes, dropout=0.3, supports=None, supports_cluster=None,
                 transmit=None, length=12,
                 in_dim=1, in_dim_cluster=1, out_dim=1, residual_channels=64, dilation_channels=64,
                 skip_channels=256, end_channels=512):
        super(PGNN, self).__init__(num_nodes,in_dim,out_dim)
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.transmit = transmit
        self.cluster_nodes = cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                            out_channels=residual_channels,
                                            kernel_size=(1, 1))
        self.supports = supports
        self.supports_cluster = supports_cluster

        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster += len(supports_cluster)

        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster = Parameter(torch.zeros(cluster_nodes, cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len += 1
        self.supports_len_cluster += 1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10, cluster_nodes).to(device), requires_grad=True).to(device)

        self.block1 = GCNPool(2 * dilation_channels, dilation_channels, num_nodes, length - 6, 3, dropout, num_nodes,
                              self.supports_len)
        self.block2 = GCNPool(2 * dilation_channels, dilation_channels, num_nodes, length - 9, 2, dropout, num_nodes,
                              self.supports_len)

        self.block_cluster1 = GCNPool(dilation_channels, dilation_channels, cluster_nodes, length - 6, 3, dropout,
                                      cluster_nodes,
                                      self.supports_len)
        self.block_cluster2 = GCNPool(dilation_channels, dilation_channels, cluster_nodes, length - 9, 2, dropout,
                                      cluster_nodes,
                                      self.supports_len)

        self.skip_conv1 = Conv2d(2 * dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(2 * dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.bn = BatchNorm2d(in_dim, affine=False)
        self.conv_cluster1 = Conv2d(dilation_channels, out_dim, kernel_size=(1, 3),
                                    stride=(1, 1), bias=True)
        self.bn_cluster = BatchNorm2d(in_dim_cluster, affine=False)

        self.transmit1 = Transmit(dilation_channels, length, transmit, num_nodes, cluster_nodes)
        self.transmit2 = Transmit(dilation_channels, length - 6, transmit, num_nodes, cluster_nodes)
        self.transmit3 = Transmit(dilation_channels, length - 9, transmit, num_nodes, cluster_nodes)

    def forward(self, input, input_cluster):
        input=input.transpose(1, 3)
        input_cluster = input_cluster.transpose(1, 3)
        x = self.bn(input)
        input_c = input_cluster
        
        x_cluster = self.bn_cluster(input_c)
    
        # nodes
        A = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        d = 1 / (torch.sum(A, -1))
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)

        new_supports = self.supports + [A]
        # region
        A_cluster = F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
        d_c = 1 / (torch.sum(A_cluster, -1))
        D_c = torch.diag_embed(d_c)
        A_cluster = torch.matmul(D_c, A_cluster)

        new_supports_cluster = self.supports_cluster + [A_cluster]

        # network

        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)

        transmit1 = self.transmit1(x, x_cluster)
        x_1 = (torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster))
        x = torch.cat((x, x_1), 1)
        # x = torch.cat((x, x_cluster), 1)

        skip = 0
        # 1
        x_cluster = self.block_cluster1(x_cluster, new_supports_cluster)
        x = self.block1(x, new_supports)

        transmit2 = self.transmit2(x, x_cluster)
        x_2 = (torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster))
        x = torch.cat((x, x_2), 1)
        # x = torch.cat((x, x_cluster), 1)

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # 2
        x_cluster = self.block_cluster2(x_cluster, new_supports_cluster)
        x = self.block2(x, new_supports)

        # transmit3 = self.transmit3(x, x_cluster)
        # x_3 = (torch.einsum('bmn,bcnl->bcml', transmit3, x_cluster))
        # x = torch.cat((x, x_3), 1)
        x = torch.cat((x, x_cluster), 1)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        print(x.shape)
        exit()

        return x

class GCNPool(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2):
        super(GCNPool, self).__init__()
        self.time_conv = Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_time(c_out, 2 * c_out, Kt, dropout, support_len, order)

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        self.TAT = TATT_1(c_out, num_nodes, tem_size)
        self.c_out = c_out
        # self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, x, support):
        residual = self.conv1(x)

        x = self.time_conv(x)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        x = self.multigcn(x, support)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)

        T_coef = self.TAT(x)
        T_coef = T_coef.transpose(-1, -2)
        x = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        out = self.bn(x + residual[:, :, :, -x.size(3):])
        return out

class Transmit(nn.Module):
    def __init__(self, c_in, tem_size, transmit, num_nodes, cluster_nodes):
        super(Transmit, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, cluster_nodes), requires_grad=True)
        self.c_in = c_in
        self.transmit = transmit

    def forward(self, seq, seq_cluster):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  # b,n,l

        c2 = seq_cluster.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)

        coefs = logits# * self.transmit
        return coefs

class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        # print(c2.shape)
        f2 = self.conv2(c2).squeeze()  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs

class multi_gcn_time(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_time,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class linear_time(nn.Module):
    def __init__(self,c_in,c_out,Kt):
        super(linear_time,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
