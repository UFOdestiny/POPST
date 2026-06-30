"""STLLM7: per-mode embedding + correlation-graph propagation across modes.

Where the original STLLM7 mixed M raw-scalar mode values once with a
learned dynamic adjacency, this redesign keeps mode identity in a
d_pm-dim per-mode subspace and propagates information across modes via a
graph whose edges are an interpretable blend of:
  (1) a learned static prior,
  (2) the empirical Pearson correlation matrix of the input window
      (data-driven, sample-conditional), and
  (3) a content-derived dynamic adjacency from the per-sample mean.
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


def pearson_corr_per_node(x, eps=1e-5):
    """Per-sample, *per-node* Pearson correlation across the M modes, over the T window.

    Args:
        x: (B, T, N, M)
    Returns:
        (B, N, M, M) correlation matrix for every (sample, node).
    """
    B, T, N, M = x.shape
    flat = x.permute(0, 2, 3, 1)                       # (B, N, M, T)
    flat = flat - flat.mean(dim=-1, keepdim=True)
    std = flat.std(dim=-1, keepdim=True).clamp_min(eps)
    flat = flat / std
    corr = (flat @ flat.transpose(-1, -2)) / max(T - 1, 1)
    return corr                                        # (B, N, M, M)


class ContextConditionalGraphMixer(nn.Module):
    """Cross-mode propagation on a *context-conditional* coupling graph.

    The data analysis (data/Analysis/cross_mode_coupling.ipynb) shows that
    cross-mode coupling is strong globally but highly heterogeneous across
    space (hubs couple ~2-6x more than the periphery) and time. A single
    global coupling matrix is therefore mis-specified. This mixer builds a
    *per-node* adjacency `A[b, n] (M, M)` as an interpretable blend of four
    sources, so the model can learn where and when modes couple:

      (1) static    — a global learned prior shared by all nodes;
      (2) pearson    — the per-node empirical Pearson correlation of the
                       raw input window (data-driven, sample- & node-conditional);
      (3) node       — a node-identity coupling bias from the node embedding
                       (captures persistent place structure: hub vs periphery);
      (4) dynamic    — a per-node, content-derived adjacency from the local
                       pooled signal (captures the current context, e.g. rush).

    The four are combined on the simplex via learned `mix_logits` (logged for
    interpretation), and message passing uses the resulting per-node graph.
    """

    def __init__(self, num_modes, d_pm, d_node, hidden_dim=16, dropout=0.1, use_pearson=True):
        super().__init__()
        self.num_modes = num_modes
        self.use_pearson = use_pearson
        self.norm = RMSNorm(d_pm)
        self.static_logits = nn.Parameter(torch.zeros(num_modes, num_modes))
        # Node-identity coupling: node embedding -> per-node (M, M) logits.
        self.node_proj = nn.Linear(d_node, num_modes * num_modes)
        # Content dynamic: per-node pooled features -> per-node (M, M) logits.
        self.dynamic_proj = nn.Sequential(
            nn.Linear(num_modes * d_pm, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modes * num_modes),
        )
        # Four log-mix weights [static, pearson, node, dynamic] on the simplex.
        # Init prefers pearson (encoder) or node-identity (decoder, no pearson).
        init = torch.tensor([0.0, 1.0 if use_pearson else -1e9, 0.5, 0.0])
        self.mix_logits = nn.Parameter(init)
        self.value_proj = nn.Linear(d_pm, d_pm)
        self.out_proj = nn.Linear(d_pm, d_pm)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(d_pm))
        # per-node raw-input pearson computed once per forward; cached.
        self._pearson = None

    def set_pearson(self, x_raw):
        """Cache the per-node Pearson correlation of the *raw* input (B, T, N, M)."""
        self._pearson = pearson_corr_per_node(x_raw)   # (B, N, M, M)

    def forward(self, x, node_emb):
        # x: (B, T, N, M, d_pm); node_emb: (N, d_node); _pearson: (B, N, M, M).
        residual = x
        x_n = self.norm(x)
        B, T, N, M, _ = x_n.shape

        static_adj = F.softmax(self.static_logits, dim=-1)             # (M, M)
        static_adj = static_adj.view(1, 1, M, M)                       # broadcast over (B, N)

        # Node-identity coupling (B-broadcast, per-node).
        node_logits = self.node_proj(node_emb).view(1, N, M, M)
        node_adj = F.softmax(node_logits, dim=-1)

        if self._pearson is not None:
            pearson_adj = F.softmax(self._pearson, dim=-1)             # (B, N, M, M)
        else:
            pearson_adj = static_adj

        # Per-node content pooling over the T window keeps node identity.
        pooled = x_n.mean(dim=1).reshape(B, N, M * x_n.shape[-1])      # (B, N, M*d_pm)
        dyn_adj = F.softmax(self.dynamic_proj(pooled).view(B, N, M, M), dim=-1)

        w = F.softmax(self.mix_logits, dim=0)
        adj = w[0] * static_adj + w[1] * pearson_adj + w[2] * node_adj + w[3] * dyn_adj  # (B,N,M,M)

        v = self.value_proj(x_n)                                       # (B, T, N, M, d_pm)
        message = torch.einsum("bnij,btnjd->btnid", adj, v)
        message = self.dropout(self.out_proj(message))
        return residual + self.gate * message


class STLLM7Block(nn.Module):
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
    """STLLM7: per-mode embedding + context-conditional cross-mode coupling graph.

    The cross-mode coupling graph is conditioned on node identity and local
    content (see ContextConditionalGraphMixer), so coupling adapts across space
    and time instead of being a single global rule.
    """

    intro = "STLLM7: per-mode embedding with a context-conditional (static/Pearson/node/dynamic) coupling graph bracketing the ST backbone."

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        d_ff=384,
        num_layers=4,
        d_pm=16,
        mode_hidden_dim=16,
        dropout=0.1,
        **args,
    ):
        super().__init__(**args)
        self.d_model = d_model
        self.d_pm = d_pm
        self.num_modes = self.input_dim

        self.node_embedding = nn.Embedding(self.node_num, d_model)

        self.mode_embed = PerModeEmbedding(self.num_modes, d_pm)
        self.encoder_mixer = ContextConditionalGraphMixer(
            self.num_modes, d_pm, d_node=d_model, hidden_dim=mode_hidden_dim,
            dropout=dropout, use_pearson=True,
        )
        # Decoder mixer cannot reuse the input Pearson because its tokens have d_pm features only.
        self.decoder_mixer_pre = ContextConditionalGraphMixer(
            self.num_modes, d_pm, d_node=d_model, hidden_dim=mode_hidden_dim,
            dropout=dropout, use_pearson=False,
        )
        self.mode_fuser = ModeFuser(self.num_modes, d_pm, d_model)

        self.blocks = nn.ModuleList(
            [STLLM7Block(d_model, num_heads, d_ff, self.seq_len, dropout) for _ in range(num_layers)]
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

        node_ids = torch.arange(node_num, device=x.device)
        node_emb = self.node_embedding(node_ids)     # (N, d_model)

        # Pearson is computed on the raw mode signal so it stays unbiased by learned scales.
        with torch.no_grad():
            self.encoder_mixer.set_pearson(x)

        x = self.mode_embed(x)
        x = self.encoder_mixer(x, node_emb)          # node-conditional coupling graph
        x = self.mode_fuser(x)

        x = x + node_emb.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        token = x[:, -1, :, :]                       # (B, N, d_model)
        modes = self.mode_unfuser(token)             # (B, N, M, d_pm)
        # Decoder graph mixer expects a (B, T, N, M, d_pm) shape — unsqueeze a length-1 T axis.
        modes = self.decoder_mixer_pre(modes.unsqueeze(1), node_emb).squeeze(1)
        out = self.head(modes)                       # (B, N, M, H)
        return out.permute(0, 3, 1, 2).contiguous()  # (B, H, N, M)
