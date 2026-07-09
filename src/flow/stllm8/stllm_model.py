"""STLLM8: per-mode embedding + low-rank cross-mode bilinear mixer.

The original STLLM8 mixed M raw scalar mode values with a softmax
relation matrix and a Hadamard cross term, then collapsed them with a
single Linear(M, d_model). This redesign keeps each mode in its own
d_pm-dim subspace, then runs a low-rank bilinear cross-mode mixer that
combines

  msg_i  = sum_j  A_ij * (W_v x_j)        (additive correlation)
  cross  = (W_l x_i) ⊙ (W_r msg_i)        (multiplicative interaction)

with low-rank per-mode projections so the parameter count stays small.
The mixer is applied at both the encoder side (after embedding) and the
decoder side (before the per-mode forecast head).
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


class PerModeReadout(nn.Module):
    """Per-mode scalar readout for horizon-specific tokens: (B,H,N,M,d_pm)->(B,H,N,M)."""

    def __init__(self, num_modes, d_pm):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_modes, d_pm))
        self.bias = nn.Parameter(torch.zeros(num_modes))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return torch.einsum("bhnmd,md->bhnm", x, self.weight) + self.bias


class HorizonQueryDecoder(nn.Module):
    """Query-based multi-horizon decoder (see crossmode_availability.ipynb).

    Replaces the single last-step token (shared across all forecast steps) with one
    learned step embedding per forecast step that modulates a query cross-attending
    over the full encoded sequence and is added to the output token, so cross-modal
    signal is realized at every horizon. A last-step residual keeps h=1 near original.
    """

    def __init__(self, d_model, horizon, num_heads, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        if d_model % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.step_emb = nn.Parameter(torch.empty(horizon, d_model))
        nn.init.normal_(self.step_emb, std=0.02)
        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        H, hd, nh = self.horizon, self.head_dim, self.num_heads
        seq = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        seq_n = self.norm(seq)
        q_in = seq_n[:, -1:, :] + self.step_emb.unsqueeze(0)
        q = self.q_proj(q_in).view(B * N, H, nh, hd).transpose(1, 2)
        k = self.k_proj(seq_n).view(B * N, T, nh, hd).transpose(1, 2)
        v = self.v_proj(seq_n).view(B * N, T, nh, hd).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B * N, H, D)
        tokens = self.out_proj(ctx) + seq[:, -1:, :] + self.step_emb.unsqueeze(0)
        return tokens.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()


class LowRankBilinearMixer(nn.Module):
    """Cross-mode bilinear mixer with low-rank per-mode projections."""

    def __init__(self, num_modes, d_pm, rank=8, dropout=0.1):
        super().__init__()
        self.num_modes = num_modes
        self.d_pm = d_pm
        self.rank = min(rank, d_pm)
        self.norm = RMSNorm(d_pm)
        self.relation_logits = nn.Parameter(torch.zeros(num_modes, num_modes))
        # Low-rank per-mode value, left, and right projections.
        self.W_value_in = nn.Parameter(torch.empty(num_modes, d_pm, self.rank))
        self.W_value_out = nn.Parameter(torch.empty(num_modes, self.rank, d_pm))
        self.W_left = nn.Parameter(torch.empty(num_modes, d_pm, self.rank))
        self.W_right = nn.Parameter(torch.empty(num_modes, d_pm, self.rank))
        self.W_cross_out = nn.Parameter(torch.empty(num_modes, self.rank, d_pm))
        self.msg_gain = nn.Parameter(torch.full((d_pm,), 1e-3))
        self.cross_gain = nn.Parameter(torch.full((d_pm,), 1e-3))
        self.gate = nn.Parameter(torch.zeros(d_pm))
        self.dropout = nn.Dropout(dropout)
        for w in (self.W_value_in, self.W_value_out, self.W_left, self.W_right, self.W_cross_out):
            nn.init.normal_(w, std=0.02)

    def forward(self, x):
        # x: (..., M, d_pm).
        residual = x
        x_n = self.norm(x)
        adj = F.softmax(self.relation_logits, dim=-1)                          # (M, M)

        v_low = torch.einsum("...md,mdr->...mr", x_n, self.W_value_in)         # (..., M, r)
        message_low = torch.einsum("ij,...jr->...ir", adj, v_low)              # (..., M, r)
        message = torch.einsum("...mr,mrd->...md", message_low, self.W_value_out)

        left_low = torch.einsum("...md,mdr->...mr", x_n, self.W_left)
        right_low = torch.einsum("...md,mdr->...mr", message, self.W_right)
        cross_low = left_low * right_low
        cross = torch.einsum("...mr,mrd->...md", cross_low, self.W_cross_out)

        mixed = self.msg_gain * message + self.cross_gain * cross
        mixed = self.dropout(mixed)
        return residual + self.gate * mixed


class STLLM8Block(nn.Module):
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
    """STLLM8: per-mode embedding + low-rank cross-mode bilinear mixer."""

    intro = "STLLM8: per-mode embedding with a low-rank cross-mode bilinear mixer bracketing the ST backbone."

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        d_pm=16,
        mode_rank=8,
        reduction_ratio=4,  # kept for backward-compat CLI; unused
        dropout=0.1,
        **args,
    ):
        super().__init__(**args)
        self.d_model = d_model
        self.d_pm = d_pm
        self.num_modes = self.input_dim

        self.mode_embed = PerModeEmbedding(self.num_modes, d_pm)
        self.encoder_mixer = LowRankBilinearMixer(self.num_modes, d_pm, rank=mode_rank, dropout=dropout)
        self.decoder_mixer = LowRankBilinearMixer(self.num_modes, d_pm, rank=mode_rank, dropout=dropout)
        self.mode_fuser = ModeFuser(self.num_modes, d_pm, d_model)

        self.node_embedding = nn.Embedding(self.node_num, d_model)
        self.blocks = nn.ModuleList(
            [STLLM8Block(d_model, num_heads, d_ff, self.seq_len, dropout) for _ in range(num_layers)]
        )

        self.horizon_decoder = HorizonQueryDecoder(d_model, self.horizon, num_heads, dropout)
        self.mode_unfuser = ModeUnfuser(self.num_modes, d_pm, d_model)
        self.head = PerModeReadout(self.num_modes, d_pm)
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

        x = self.mode_embed(x)
        x = self.encoder_mixer(x)
        x = self.mode_fuser(x)

        node_ids = torch.arange(node_num, device=x.device)
        x = x + self.node_embedding(node_ids).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        # Query-based multi-horizon decode: one token per forecast step.
        tokens = self.horizon_decoder(x)      # (B, H, N, d_model)
        modes = self.mode_unfuser(tokens)     # (B, H, N, M, d_pm)
        modes = self.decoder_mixer(modes)
        out = self.head(modes)                # (B, H, N, M)
        return out.contiguous()               # (B, H, N, M)
