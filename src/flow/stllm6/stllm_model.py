"""STLLM6: per-mode embedding + multi-head cross-mode self-attention.

The original STLLM6 mixed the M raw mobility modes once before flattening
them into the d_model channel space, after which the temporal/spatial
backbone could no longer reason about which feature belonged to which
mode. This redesign keeps mode identity explicit by giving each mode its
own embedding vector and runs a multi-head cross-mode self-attention
mixer on the per-mode tokens at BOTH the encoder side (after embedding)
and the decoder side (before the per-mode head). The backbone is kept
untouched.
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
    """Lift each scalar mode value into its own d_pm-dim subspace."""

    def __init__(self, num_modes, d_pm):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_modes, d_pm))
        self.bias = nn.Parameter(torch.zeros(num_modes, d_pm))
        self.mode_emb = nn.Parameter(torch.zeros(num_modes, d_pm))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.mode_emb, std=0.02)

    def forward(self, x):
        # x: (B, T, N, M) -> (B, T, N, M, d_pm)
        x = x.unsqueeze(-1) * self.weight + self.bias
        return x + self.mode_emb


class ModeFuser(nn.Module):
    """Concatenate per-mode features and project into the d_model space."""

    def __init__(self, num_modes, d_pm, d_model):
        super().__init__()
        self.norm = RMSNorm(num_modes * d_pm)
        self.proj = nn.Linear(num_modes * d_pm, d_model)

    def forward(self, x):
        # x: (..., M, d_pm) -> (..., d_model)
        flat = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        return self.proj(self.norm(flat))


class ModeUnfuser(nn.Module):
    """Project a d_model token back into per-mode (M, d_pm) features."""

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
    """Independent linear head per mode predicting `horizon` steps."""

    def __init__(self, num_modes, d_pm, horizon):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_modes, d_pm, horizon))
        self.bias = nn.Parameter(torch.zeros(num_modes, horizon))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # x: (B, N, M, d_pm) -> (B, N, M, horizon)
        return torch.einsum("bnmd,mdh->bnmh", x, self.weight) + self.bias


class CrossModeAttention(nn.Module):
    """Multi-head self-attention across modes; modes are tokens of dim d_pm."""

    def __init__(self, d_pm, num_heads=2, dropout=0.1):
        super().__init__()
        if d_pm % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = d_pm // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = RMSNorm(d_pm)
        self.qkv = nn.Linear(d_pm, 3 * d_pm)
        self.out_proj = nn.Linear(d_pm, d_pm)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(d_pm))

    def forward(self, x):
        # x: (..., M, d_pm)
        residual = x
        x_n = self.norm(x)
        lead = x_n.shape[:-2]
        M, D = x_n.shape[-2], x_n.shape[-1]
        qkv = self.qkv(x_n).reshape(*lead, M, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=-3)
        # Bring head dim before mode for attention.
        q = q.transpose(-2, -3)  # (..., heads, M, head_dim)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (..., heads, M, head_dim)
        out = out.transpose(-2, -3).reshape(*lead, M, D)
        return residual + self.gate * self.out_proj(out)


class STLLM6Block(nn.Module):
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
    """STLLM6: per-mode embedding + cross-mode self-attention at encoder and decoder."""

    intro = "STLLM6: per-mode embedding with cross-mode multi-head self-attention bracketing the ST backbone."

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        d_pm=16,
        mode_attn_heads=2,
        mode_attn_dim=16,  # kept for backward-compat CLI; unused
        dropout=0.1,
        **args,
    ):
        super().__init__(**args)
        self.d_model = d_model
        self.d_pm = d_pm
        self.num_modes = self.input_dim

        self.mode_embed = PerModeEmbedding(self.num_modes, d_pm)
        self.encoder_mixer = CrossModeAttention(d_pm, num_heads=mode_attn_heads, dropout=dropout)
        self.decoder_mixer = CrossModeAttention(d_pm, num_heads=mode_attn_heads, dropout=dropout)
        self.mode_fuser = ModeFuser(self.num_modes, d_pm, d_model)

        self.node_embedding = nn.Embedding(self.node_num, d_model)
        self.blocks = nn.ModuleList(
            [STLLM6Block(d_model, num_heads, d_ff, self.seq_len, dropout) for _ in range(num_layers)]
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

    def forward(self, x, label=None):
        batch_size, _, node_num, _ = x.shape

        # Per-mode encoding with cross-mode attention.
        x = self.mode_embed(x)               # (B, T, N, M, d_pm)
        x = self.encoder_mixer(x)
        x = self.mode_fuser(x)               # (B, T, N, d_model)

        node_ids = torch.arange(node_num, device=x.device)
        x = x + self.node_embedding(node_ids).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        # Decode the last step in per-mode subspace with another cross-mode mixer.
        token = x[:, -1, :, :]               # (B, N, d_model)
        modes = self.mode_unfuser(token)     # (B, N, M, d_pm)
        modes = self.decoder_mixer(modes)
        out = self.head(modes)               # (B, N, M, H)
        return out.permute(0, 3, 1, 2).contiguous()  # (B, H, N, M)
