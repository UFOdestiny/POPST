import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


def _mix_support(adaptive_support, support_prior, prior_gate):
    if support_prior is None:
        return adaptive_support

    gate = torch.sigmoid(prior_gate)
    mixed = gate * adaptive_support + (1.0 - gate) * support_prior
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class FusionAVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, support_prior=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        if support_prior is None:
            self.register_buffer("support_prior", None)
        else:
            self.register_buffer("support_prior", support_prior)
        self.prior_gate = nn.Parameter(torch.tensor(0.0))

        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def _build_supports(self, node_embed):
        node_num = node_embed.shape[0]
        adaptive_support = F.softmax(
            F.relu(torch.mm(node_embed, node_embed.transpose(0, 1))), dim=-1
        )
        fused_support = _mix_support(
            adaptive_support, self.support_prior, self.prior_gate
        )

        support_set = [torch.eye(node_num, device=node_embed.device), fused_support]
        for _ in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * fused_support, support_set[-1]) - support_set[-2]
            )
        return torch.stack(support_set, dim=0)

    def forward(self, x, node_embed):
        supports = self._build_supports(node_embed)
        weights = torch.einsum("nd,dkio->nkio", node_embed, self.weights_pool)
        bias = torch.matmul(node_embed, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class FMGCNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, support_prior=None):
        super().__init__()
        self.hidden_dim = dim_out
        self.gate = FusionAVWGCN(
            dim_in + self.hidden_dim,
            2 * dim_out,
            cheb_k,
            embed_dim,
            support_prior=support_prior,
        )
        self.update = FusionAVWGCN(
            dim_in + self.hidden_dim,
            dim_out,
            cheb_k,
            embed_dim,
            support_prior=support_prior,
        )

    def forward(self, x, state, node_embed):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embed))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embed))
        return r * state + (1.0 - r) * hc

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)


class FMGCNEncoder(nn.Module):
    def __init__(
        self, dim_in, dim_out, cheb_k, embed_dim, num_layer, support_prior=None
    ):
        super().__init__()
        assert num_layer >= 1, "At least one recurrent layer is required."
        self.num_layer = num_layer
        self.cells = nn.ModuleList(
            [
                FMGCNCell(
                    dim_in if i == 0 else dim_out,
                    dim_out,
                    cheb_k,
                    embed_dim,
                    support_prior=support_prior,
                )
                for i in range(num_layer)
            ]
        )

    def forward(self, x, init_state, node_embed):
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for layer_idx in range(self.num_layer):
            state = init_state[layer_idx]
            inner_states = []
            for t in range(seq_length):
                state = self.cells[layer_idx](
                    current_inputs[:, t, :, :], state, node_embed
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, node_num):
        init_states = [
            cell.init_hidden_state(batch_size, node_num) for cell in self.cells
        ]
        return torch.stack(init_states, dim=0)


class FMGCN(BaseModel):
    """AGCRN-inspired standalone backbone with fused static/adaptive graph supports."""

    def __init__(
        self,
        embed_dim,
        rnn_unit,
        num_layer,
        cheb_k,
        dropout=0.1,
        support_prior=None,
        **args,
    ):
        super().__init__(**args)
        self.node_embed = nn.Parameter(
            torch.randn(self.node_num, embed_dim), requires_grad=True
        )
        self.encoder = FMGCNEncoder(
            self.input_dim,
            rnn_unit,
            cheb_k,
            embed_dim,
            num_layer,
            support_prior=support_prior,
        )
        self.readout = nn.Sequential(
            nn.Linear(rnn_unit, rnn_unit),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.end_proj = nn.Linear(rnn_unit, self.horizon * self.output_dim)

    def forward(self, source, label=None):
        batch_size, _, node_num, _ = source.shape
        init_state = self.encoder.init_hidden(batch_size, node_num).to(source.device)
        output, _ = self.encoder(source, init_state, self.node_embed)
        output = self.readout(output[:, -1, :, :])
        pred = self.end_proj(output)
        pred = pred.view(batch_size, node_num, self.horizon, self.output_dim)
        return pred.permute(0, 2, 1, 3).contiguous()
