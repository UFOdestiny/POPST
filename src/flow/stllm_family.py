"""Shared STLLM-family building blocks and controlled add/remove variants."""

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


class SpatialAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, node_num, d_model = x.shape

        q = self.q_proj(x).reshape(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, node_num, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(batch_size, node_num, d_model)
        return self.out_proj(out)


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


class STLLMFamilyBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1, enable_spatial=True, gated_fusion=False):
        super().__init__()
        self.enable_spatial = enable_spatial
        self.gated_fusion = gated_fusion

        self.temporal_norm = RMSNorm(d_model)
        self.temporal_attn = TemporalAttention(d_model, num_heads, max_seq_len, dropout)

        if enable_spatial:
            self.spatial_norm = RMSNorm(d_model)
            self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

        if gated_fusion:
            self.temporal_gate = nn.Linear(d_model, d_model)
            if enable_spatial:
                self.spatial_gate = nn.Linear(d_model, d_model)

    def _apply_residual(self, residual, update, gate_layer=None):
        if gate_layer is not None:
            update = update * torch.sigmoid(gate_layer(residual))
        return residual + self.dropout(update)

    def forward(self, x):
        batch_size, seq_len, node_num, d_model = x.shape

        x_reshape = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_attn(x_norm)
        gate_layer = self.temporal_gate if self.gated_fusion else None
        x_reshape = self._apply_residual(x_reshape, temporal_out, gate_layer)
        x = x_reshape.reshape(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)

        if self.enable_spatial:
            x_reshape = x.contiguous().reshape(batch_size * seq_len, node_num, d_model)
            x_norm = self.spatial_norm(x_reshape)
            spatial_out = self.spatial_attn(x_norm)
            gate_layer = self.spatial_gate if self.gated_fusion else None
            x_reshape = self._apply_residual(x_reshape, spatial_out, gate_layer)
            x = x_reshape.reshape(batch_size, seq_len, node_num, d_model)

        x_reshape = x.reshape(batch_size * seq_len * node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        ffn_out = self.ffn(x_norm)
        x_reshape = x_reshape + ffn_out
        x = x_reshape.reshape(batch_size, seq_len, node_num, d_model)

        return x


class STLLMVariant(BaseModel):
    intro = "STLLM variant"

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        dropout=0.1,
        *,
        add_time_embedding=False,
        gated_fusion=False,
        enable_spatial=True,
        add_node_embedding=True,
        use_output_norm=True,
        **args,
    ):
        super().__init__(**args)
        self.d_model = d_model
        self.add_time_embedding = add_time_embedding
        self.add_node_embedding = add_node_embedding
        self.use_output_norm = use_output_norm

        self.input_embedding = nn.Linear(self.input_dim, d_model)

        if add_time_embedding:
            self.time_embedding = nn.Parameter(torch.empty(1, self.seq_len, 1, d_model))

        if add_node_embedding:
            self.node_embedding = nn.Embedding(self.node_num, d_model)

        self.blocks = nn.ModuleList(
            [
                STLLMFamilyBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    self.seq_len,
                    dropout,
                    enable_spatial=enable_spatial,
                    gated_fusion=gated_fusion,
                )
                for _ in range(num_layers)
            ]
        )

        if use_output_norm:
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

        if self.add_time_embedding:
            nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, x, label=None):
        batch_size, seq_len, node_num, _ = x.shape

        x = self.input_embedding(x)

        if self.add_time_embedding:
            x = x + self.time_embedding[:, :seq_len, :, :]

        if self.add_node_embedding:
            node_ids = torch.arange(node_num, device=x.device)
            node_emb = self.node_embedding(node_ids)
            x = x + node_emb.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        if self.use_output_norm:
            x = self.output_norm(x)

        x = x[:, -1, :, :]
        x = self.output_proj(x)
        x = x.reshape(batch_size, node_num, self.horizon, self.output_dim)
        return x.permute(0, 2, 1, 3)


class STLLM2Model(STLLMVariant):
    """Compared with STLLM, STLLM2 only adds a learnable time embedding."""

    intro = "Compared with STLLM, STLLM2 only adds a learnable time embedding."

    def __init__(self, **args):
        super().__init__(add_time_embedding=True, **args)


class STLLM3Model(STLLMVariant):
    """Compared with STLLM, STLLM3 adds gated fusion on temporal and spatial residuals."""

    intro = "Compared with STLLM, STLLM3 adds gated fusion on temporal and spatial residuals."

    def __init__(self, **args):
        super().__init__(gated_fusion=True, **args)


class STLLM4Model(STLLMVariant):
    """Compared with STLLM, STLLM4 removes the spatial attention branch."""

    intro = "Compared with STLLM, STLLM4 removes the spatial attention branch."

    def __init__(self, **args):
        super().__init__(enable_spatial=False, **args)


class STLLM5Model(STLLMVariant):
    """Compared with STLLM, STLLM5 removes node embedding and the output RMSNorm."""

    intro = "Compared with STLLM, STLLM5 removes node embedding and the output RMSNorm."

    def __init__(self, **args):
        super().__init__(add_node_embedding=False, use_output_norm=False, **args)
