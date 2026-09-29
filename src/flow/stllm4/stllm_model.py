"""STLLM4 keeps the STLLM backbone but removes the spatial attention branch.

Compared with the original STLLM, each block replaces spatial attention with a
lightweight global node mixer while keeping temporal attention and the feed-forward path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x, seq_len)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GlobalSpatialMixer(nn.Module):
    """Global node mixer with mean + max + gated content context.

    A single mean-pool collapses node heterogeneity into one signal per
    batch token. Adding a max-pool keeps the most-active node's signature,
    and a learned content-conditioned gate decides per-node how much of
    the global context to inject.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 3 streams: per-node feature, mean-pool context, max-pool context.
        self.in_proj = nn.Linear(d_model * 3, d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        # Per-node, content-conditioned gate so heterogeneous nodes can
        # selectively absorb the global summary.
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-3))

    def forward(self, x):
        mean_ctx = x.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        max_ctx = x.amax(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        fused = torch.cat([x, mean_ctx, max_ctx], dim=-1)
        update = self.out_proj(F.gelu(self.in_proj(fused)))
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(update * gate * self.layer_scale)


class STLLM4Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_attn = TemporalAttention(d_model, num_heads, max_seq_len, dropout)
        self.spatial_norm = RMSNorm(d_model)
        self.spatial_mixer = GlobalSpatialMixer(d_model, d_ff, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, node_num, d_model = x.shape

        x_reshape = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_attn(x_norm)
        x_reshape = x_reshape + self.dropout(temporal_out)
        x = x_reshape.reshape(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)

        x_reshape = x.contiguous().reshape(batch_size * seq_len, node_num, d_model)
        x_norm = self.spatial_norm(x_reshape)
        spatial_out = self.spatial_mixer(x_norm)
        x = (x_reshape + spatial_out).reshape(batch_size, seq_len, node_num, d_model)

        x_reshape = x.reshape(batch_size * seq_len * node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        ffn_out = self.ffn(x_norm)
        x_reshape = x_reshape + ffn_out
        return x_reshape.reshape(batch_size, seq_len, node_num, d_model)


class STLLM(BaseModel):
    """Compared with STLLM, STLLM4 swaps spatial attention for a global node mixer."""

    intro = "Compared with STLLM, STLLM4 swaps spatial attention for a global node mixer."

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        dropout=0.1,
        **args,
    ):
        super().__init__(**args)
        self.d_model = d_model

        self.input_embedding = nn.Linear(self.input_dim, d_model)
        self.node_embedding = nn.Embedding(self.node_num, d_model)
        self.blocks = nn.ModuleList(
            [STLLM4Block(d_model, num_heads, d_ff, self.seq_len, dropout) for _ in range(num_layers)]
        )
        self.output_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.output_dim * self.horizon)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, label=None):
        batch_size, _, node_num, _ = x.shape

        x = self.input_embedding(x)

        node_ids = torch.arange(node_num, device=x.device)
        node_emb = self.node_embedding(node_ids)
        x = x + node_emb.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        x = self.output_norm(x)
        x = x[:, -1, :, :]
        x = self.output_proj(x)
        x = x.reshape(batch_size, node_num, self.horizon, self.output_dim)
        return x.permute(0, 2, 1, 3)
