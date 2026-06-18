"""STLLM9: per-mode embedding + token-conditioned cross-mode router.

The original STLLM9 routed M scalar mode values with a token-conditioned
softmax relation matrix and a single Linear(M, d_model) bottleneck. This
redesign keeps each mode in a d_pm-dim subspace and routes information
across modes with attention whose query/key are mode-signature vectors
modulated by per-token content. Concretely, for each token (b, t, n) the
router constructs

  q_m = (q_m^proto) ⊙ (1 + tanh α_m(token))
  k_m = (k_m^proto) ⊙ (1 + tanh β_m(token))

so that the cross-mode adjacency adapts to what each token currently
encodes. The mixer is applied at both the encoder side (after embedding)
and the decoder side (before the per-mode forecast head).
"""

import math
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
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
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


class PerModeEmbedding(nn.Module):
    def __init__(self, num_modes, d_pm):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_modes, d_pm))
        self.bias = nn.Parameter(torch.zeros(num_modes, d_pm))
        self.mode_emb = nn.Parameter(torch.zeros(num_modes, d_pm))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.mode_emb, std=0.02)

    def forward(self, x):
        x = x.unsqueeze(-1) * self.weight + self.bias
        return x + self.mode_emb


class ModeFuser(nn.Module):
    def __init__(self, num_modes, d_pm, d_model):
        super().__init__()
        self.norm = RMSNorm(num_modes * d_pm)
        self.proj = nn.Linear(num_modes * d_pm, d_model)

    def forward(self, x):
        flat = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        return self.proj(self.norm(flat))


class ModeUnfuser(nn.Module):
    def __init__(self, num_modes, d_pm, d_model):
        super().__init__()
        self.num_modes = num_modes
        self.d_pm = d_pm
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, num_modes * d_pm)

    def forward(self, x):
        out = self.proj(self.norm(x))
        return out.reshape(*out.shape[:-1], self.num_modes, self.d_pm)


class PerModeHead(nn.Module):
    def __init__(self, num_modes, d_pm, horizon):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_modes, d_pm, horizon))
        self.bias = nn.Parameter(torch.zeros(num_modes, horizon))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return torch.einsum("bnmd,mdh->bnmh", x, self.weight) + self.bias


class TokenConditionedRouter(nn.Module):
    """Cross-mode router whose adjacency is mode-signature × token-conditioned scale."""

    def __init__(self, num_modes, d_pm, mode_rel_dim=16, dropout=0.1):
        super().__init__()
        self.num_modes = num_modes
        self.d_pm = d_pm
        self.mode_rel_dim = mode_rel_dim
        self.norm = RMSNorm(d_pm)
        self.query_mode = nn.Parameter(torch.empty(num_modes, mode_rel_dim))
        self.key_mode = nn.Parameter(torch.empty(num_modes, mode_rel_dim))
        # Per-mode scalar modulators driven by the local token.
        self.q_scale = nn.Linear(d_pm, mode_rel_dim)
        self.k_scale = nn.Linear(d_pm, mode_rel_dim)
        self.value_proj = nn.Linear(d_pm, d_pm)
        self.out_proj = nn.Linear(d_pm, d_pm)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(d_pm))
        self.scale = math.sqrt(mode_rel_dim)
        nn.init.normal_(self.query_mode, std=0.02)
        nn.init.normal_(self.key_mode, std=0.02)

    def forward(self, x):
        # x: (..., M, d_pm).
        residual = x
        x_n = self.norm(x)

        q_mod = 1.0 + 0.1 * torch.tanh(self.q_scale(x_n))                    # (..., M, mode_rel_dim)
        k_mod = 1.0 + 0.1 * torch.tanh(self.k_scale(x_n))
        q = q_mod * self.query_mode                                          # (..., M, mode_rel_dim)
        k = k_mod * self.key_mode

        rel = torch.matmul(q, k.transpose(-2, -1)) / self.scale              # (..., M, M)
        rel = F.softmax(rel, dim=-1)
        rel = self.dropout(rel)

        v = self.value_proj(x_n)                                             # (..., M, d_pm)
        mixed = torch.matmul(rel, v)
        mixed = self.out_proj(mixed)
        return residual + self.gate * mixed


class STLLM9Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_attn = TemporalAttention(d_model, num_heads, max_seq_len, dropout)
        self.temporal_gate = nn.Linear(d_model, d_model)

        self.spatial_norm = RMSNorm(d_model)
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)
        self.spatial_gate = nn.Linear(d_model, d_model)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def _apply_residual(self, residual, gate_input, update, gate_layer):
        gate = 1.0 + 0.25 * torch.tanh(gate_layer(gate_input))
        return residual + self.dropout(update * gate)

    def forward(self, x):
        batch_size, seq_len, node_num, d_model = x.shape
        x_reshape = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * node_num, seq_len, d_model)
        x_norm = self.temporal_norm(x_reshape)
        temporal_out = self.temporal_attn(x_norm)
        x_reshape = self._apply_residual(x_reshape, x_norm, temporal_out, self.temporal_gate)
        x = x_reshape.reshape(batch_size, node_num, seq_len, d_model).permute(0, 2, 1, 3)

        x_reshape = x.contiguous().reshape(batch_size * seq_len, node_num, d_model)
        x_norm = self.spatial_norm(x_reshape)
        spatial_out = self.spatial_attn(x_norm)
        x_reshape = self._apply_residual(x_reshape, x_norm, spatial_out, self.spatial_gate)
        x = x_reshape.reshape(batch_size, seq_len, node_num, d_model)

        x_reshape = x.reshape(batch_size * seq_len * node_num, d_model)
        x_norm = self.ffn_norm(x_reshape)
        x_reshape = x_reshape + self.ffn(x_norm)
        return x_reshape.reshape(batch_size, seq_len, node_num, d_model)


class STLLM(BaseModel):
    """STLLM9: per-mode embedding + token-conditioned cross-mode router."""

    intro = "STLLM9: per-mode embedding with a token-conditioned cross-mode router bracketing the ST backbone."

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        d_pm=16,
        mode_rel_dim=16,
        dropout=0.1,
        **args,
    ):
        super().__init__(**args)
        self.d_model = d_model
        self.d_pm = d_pm
        self.num_modes = self.input_dim

        self.mode_embed = PerModeEmbedding(self.num_modes, d_pm)
        self.encoder_mixer = TokenConditionedRouter(self.num_modes, d_pm, mode_rel_dim=mode_rel_dim, dropout=dropout)
        self.decoder_mixer = TokenConditionedRouter(self.num_modes, d_pm, mode_rel_dim=mode_rel_dim, dropout=dropout)
        self.mode_fuser = ModeFuser(self.num_modes, d_pm, d_model)

        self.node_embedding = nn.Embedding(self.node_num, d_model)
        self.blocks = nn.ModuleList(
            [STLLM9Block(d_model, num_heads, d_ff, self.seq_len, dropout) for _ in range(num_layers)]
        )

        self.mode_unfuser = ModeUnfuser(self.num_modes, d_pm, d_model)
        self.head = PerModeHead(self.num_modes, d_pm, self.horizon)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.temporal_gate.weight)
            nn.init.zeros_(block.temporal_gate.bias)
            nn.init.zeros_(block.spatial_gate.weight)
            nn.init.zeros_(block.spatial_gate.bias)

        # The token-conditioned scale projections start near unity (tanh(0)=0).
        for mixer in (self.encoder_mixer, self.decoder_mixer):
            nn.init.zeros_(mixer.q_scale.weight)
            nn.init.zeros_(mixer.q_scale.bias)
            nn.init.zeros_(mixer.k_scale.weight)
            nn.init.zeros_(mixer.k_scale.bias)

    def forward(self, x, label=None):
        batch_size, _, node_num, _ = x.shape

        x = self.mode_embed(x)
        x = self.encoder_mixer(x)
        x = self.mode_fuser(x)

        node_ids = torch.arange(node_num, device=x.device)
        x = x + self.node_embedding(node_ids).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        token = x[:, -1, :, :]
        modes = self.mode_unfuser(token)
        modes = self.decoder_mixer(modes)
        out = self.head(modes)
        return out.permute(0, 3, 1, 2).contiguous()
