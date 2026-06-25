"""HealthMamba — Uncertainty-aware Spatiotemporal Graph State Space Model.

Re-implementation of the predictor in HealthMamba (IJCAI'26):
"HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model
for Effective and Reliable Healthcare Facility Visit Prediction".

The network has three components (Sec. 4 of the paper):

1. **STCE** — Unified SpatioTemporal Context Encoder.  Embeds the visit
   features, mixes them spatially with a prior-graph convolution, then
   temporally with a depthwise-conv + channel-MLP mixer, producing a
   node-time representation ``R ∈ R^{B×N×T×d_model}``.  (The framework's
   dataloader exposes only the visit tensor ``V``; the optional static
   demographics ``D`` and dynamic externals ``E`` of the paper are not in
   the pipeline, so STCE fuses the visit stream alone.)

2. **G-Mamba** — GraphMamba backbone.  A UNet over the temporal axis whose
   blocks combine *adaptive graph learning* (an attention-style data-driven
   adjacency, blended with the prior graph), graph-convolutional spatial
   mixing, a selective state-space module (Mamba) over time, and channel
   mixing — all residual.

3. **Uncertainty-aware heads** (UQ mode only) — three complementary
   mechanisms on the final representation: a *node-based* quantile head
   (ordered lo/mid/hi), a *distribution-based* Gaussian head ``(mu, logvar)``,
   and a *parameter-based* MC-dropout consistency estimate ``mc_var``.

The model has two modes, selected by the runner exactly like the other
flow models (``--cqr`` toggles the uncertainty engine):

* **Point mode** (default, ``--cqr no``).  A single regression head returns a
  point tensor ``(B, H, N, F)`` — the same contract as EnergyMamba /
  TrustEnergy — consumed by :class:`base.engine.BaseEngine`.
* **UQ mode** (``--cqr horizon`` / ``--cqr global``).  The uncertainty heads
  are built and ``forward`` returns a **dict** of normalised-space tensors

      {q_lo, q_mid, q_hi, mu, logvar, mc_var}   each (B, H, N, F)

  consumed by :class:`src.flow.healthmamba.PHQC_engine.PHQC_Engine`, which trains the joint
  objective and applies post-hoc quantile calibration with MC-dropout
  inference.  The runner signals this mode by passing ``cqr_channels`` (the
  true feature count F).

Internal tensor convention is ``(B, N, T, D)`` so the graph convolution
mixes across ``N`` at each step while the SSM scans along ``T``.

Ablation toggles (``use_stce``, ``use_gmamba``, ``use_node``, ``use_dist``,
``use_param``) reproduce the paper's ablation variants.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction)."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---------------------------------------------------------------------------
# Selective state-space module (temporal Mamba scan, per node)
# ---------------------------------------------------------------------------

class MambaSSM(nn.Module):
    """Standard selective state-space block (Mamba) over the time axis.

    Operates on ``(B2, T, D)`` with ``B2 = B*N`` — each node is an
    independent sequence.  Data-dependent ``Δ, B, C`` give the selectivity;
    implemented on top of ``mamba_ssm``'s fused ``selective_scan_fn``.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, h):  # h: (B2, T, D)
        T = h.shape[1]
        xz = self.in_proj(h)
        x, gate = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :T]
        x = F.silu(x)

        x_t = x.transpose(1, 2)
        dbc = self.x_proj(x_t)
        dt, B_ssm, C_ssm = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj(dt).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        B_ssm = B_ssm.transpose(1, 2).contiguous()
        C_ssm = C_ssm.transpose(1, 2).contiguous()

        y = selective_scan_fn(
            x.contiguous(), delta.contiguous(), A, B_ssm, C_ssm,
            self.D.float(), z=gate.transpose(1, 2).contiguous(),
            delta_bias=None, delta_softplus=True,
        )
        y = y.transpose(1, 2)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Adaptive graph learning  (Sec. 4.2.1)
# ---------------------------------------------------------------------------

class AdaptiveGraph(nn.Module):
    """Learn a data-driven, symmetric, normalised adjacency per forward pass,
    blended with the prior graph ``A0``.

    Node embeddings are pooled over time (per batch / node); an attention
    score ``a^T[W u_i ‖ W u_j]`` yields affinities, softmax-normalised, then
    symmetrised and degree-normalised.  Returns ``A* = λ A0 + (1-λ) Â``.
    """

    def __init__(self, d_model, emb_dim=32):
        super().__init__()
        self.lin = nn.Linear(d_model, emb_dim)
        self.attn = nn.Linear(2 * emb_dim, 1)
        # Learnable prior-blend weight λ ∈ (0, 1) via sigmoid.
        self.lam = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adj0):  # x: (B, N, T, D), adj0: (N, N)
        B, N, T, D = x.shape
        u = self.lin(x.mean(dim=2))                       # (B, N, e)
        ui = u.unsqueeze(2).expand(B, N, N, -1)           # (B, N, N, e)
        uj = u.unsqueeze(1).expand(B, N, N, -1)
        e = self.attn(torch.cat([ui, uj], dim=-1)).squeeze(-1)  # (B, N, N)
        e = F.leaky_relu(e, 0.2)
        a = F.softmax(e, dim=-1)

        a = 0.5 * (a + a.transpose(1, 2))                 # symmetrise
        deg = a.sum(dim=-1).clamp(min=1e-6)               # (B, N)
        dinv = deg.pow(-0.5)
        a = dinv.unsqueeze(-1) * a * dinv.unsqueeze(1)    # D^-1/2 A D^-1/2

        lam = torch.sigmoid(self.lam)
        return lam * adj0.unsqueeze(0) + (1.0 - lam) * a  # (B, N, N)


# ---------------------------------------------------------------------------
# G-Mamba block  (Sec. 4.2.2)
# ---------------------------------------------------------------------------

class GMambaBlock(nn.Module):
    """One GraphMamba block: adaptive-graph spatial mixing → SSM temporal
    mixing → channel mixing, with residual connections and a final
    projection (Eqs. 13-19 of the paper).

    ``use_graph=False`` reproduces the *w/o G-Mamba* ablation (plain Mamba
    block without the adaptive graph).
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout,
                 use_graph=True, emb_dim=32):
        super().__init__()
        self.use_graph = use_graph
        self.norm = RMSNorm(d_model)

        if use_graph:
            self.graph = AdaptiveGraph(d_model, emb_dim)
            self.gconv = nn.Linear(d_model, d_model)

        self.fwd = MambaSSM(d_model, d_state, d_conv, expand)
        self.bwd = MambaSSM(d_model, d_state, d_conv, expand)

        self.ln = nn.LayerNorm(d_model)
        self.chan_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_model, d_model),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, mamba, h):  # h: (B, N, T, D)
        B, N, T, D = h.shape
        out = mamba(h.reshape(B * N, T, D))
        return out.reshape(B, N, T, D)

    def forward(self, h, adj0):  # h: (B, N, T, D)
        hn = self.norm(h)

        # Graph-enhanced spatial mixing (residual).
        if self.use_graph:
            astar = self.graph(hn, adj0)                  # (B, N, N)
            g = torch.einsum("bnm,bmtd->bntd", astar, self.gconv(hn))
            g = F.relu(g) + h
        else:
            g = h

        # Bidirectional SSM over time (residual).
        bip = self._scan(self.fwd, g) + self._scan(self.bwd, g.flip(2)).flip(2)
        t = g + self.dropout(bip)

        # Channel mixing (residual) + projection (residual).
        c = self.chan_mlp(self.ln(t)) + t
        return self.out_proj(c) + h


# ---------------------------------------------------------------------------
# STCE  (Sec. 4.1)
# ---------------------------------------------------------------------------

class STCE(nn.Module):
    """Unified SpatioTemporal Context Encoder.

    Embeds visit features, applies a prior-graph convolution per timestep,
    a depthwise-conv + channel-MLP temporal mixer, and projects to the model
    dimension, yielding ``R ∈ R^{B×N×T×d_model}``.
    """

    def __init__(self, input_dim, d_hid, d_model, seq_len, dropout):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_hid), nn.SiLU(), nn.Dropout(dropout)
        )
        self.gconv = nn.Linear(d_hid, d_hid)
        self.depth_conv = nn.Conv1d(
            d_hid, d_hid, kernel_size=3, padding=1, groups=d_hid
        )
        self.ln = nn.LayerNorm(d_hid)
        self.chan_mlp = nn.Sequential(
            nn.Linear(d_hid, 2 * d_hid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_hid, d_hid),
        )
        self.proj = nn.Linear(d_hid, d_model)
        self.out_ln = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(dropout)
        )

    def forward(self, x, adj0):  # x: (B, N, T, F), adj0: (N, N)
        B, N, T, _ = x.shape
        h = self.embed(x)                                 # (B, N, T, d_hid)

        # Spatial: prior-graph convolution per timestep + ReLU + residual.
        g = torch.einsum("nm,bmtd->bntd", adj0, self.gconv(h))
        h = F.relu(g) + h

        # Temporal: depthwise conv over time + channel MLP (both residual).
        d = h.shape[-1]
        u = h.permute(0, 1, 3, 2).reshape(B * N, d, T)
        u = self.depth_conv(u)[..., :T].reshape(B, N, d, T).permute(0, 1, 3, 2)
        h = u + h
        h = self.chan_mlp(self.ln(h)) + h

        r = self.out_ln(self.proj(h))                     # (B, N, T, d_model)
        return self.out(r)


# ---------------------------------------------------------------------------
# HealthMamba
# ---------------------------------------------------------------------------

class HealthMamba(BaseModel):
    """GraphMamba UNet predictor with comprehensive uncertainty heads.

    Args (model-specific):
        d_model:     hidden / output embedding dimension d_model.
        d_hid:       STCE hidden dimension d_hid.
        num_layers:  G-Mamba blocks per encoder/decoder stage.
        depth:       number of encoder stages S (``depth-1`` temporal
                     down/up-sampling levels around the bottleneck).
        d_state:     SSM state dimension.
        expand:      Mamba inner-dimension expansion factor.
        d_conv:      causal-conv kernel width.
        emb_dim:     adaptive-graph node-embedding width.
        dropout:     dropout rate (also drives MC-dropout at inference).
        adj:         pre-normalised dense prior adjacency, ``(N, N)``.
        use_stce / use_gmamba / use_node / use_dist / use_param:
                     ablation toggles.
    """

    # ``--cqr`` selects PHQC_Engine (the quantile engine) and is what puts the
    # model into UQ mode; without it the model is a plain point regressor on
    # BaseEngine.  The runner's gate at this attribute only needs to *allow*
    # ``--cqr``; the actual head wiring is driven by ``cqr_channels`` below
    # (HealthMamba sizes its own UQ heads and does not consume the runner's
    # ``output_dim`` widening), so we opt into the gate without following the
    # 3*F output_dim contract that the generic CQR engine assumes.
    cqr_compatible = True

    def __init__(
        self,
        d_model,
        num_layers,
        adj,
        d_hid=128,
        depth=2,
        d_state=16,
        expand=2,
        d_conv=4,
        emb_dim=32,
        dropout=0.3,
        use_stce=True,
        use_gmamba=True,
        use_node=True,
        use_dist=True,
        use_param=True,
        cqr_channels=None,
        **args,
    ):
        super(HealthMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = max(1, num_layers)
        self.depth = max(1, depth)
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.emb_dim = emb_dim
        self.dropout_p = dropout
        self.use_stce = use_stce
        self.use_gmamba = use_gmamba
        self.use_node = use_node
        self.use_dist = use_dist
        self.use_param = use_param

        # UQ mode is driven by --cqr (PHQC_Engine).  In that mode the runner has
        # already widened output_dim to 3*F and stashed the true feature count F
        # in cqr_channels.  Without --cqr the model is a plain point regressor
        # on BaseEngine and output_dim is the true F.
        self.uq_mode = cqr_channels is not None
        self.F = cqr_channels if self.uq_mode else self.output_dim

        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer("adj", adj)

        # --- input / STCE ---
        if use_stce:
            self.stce = STCE(self.input_dim, d_hid, d_model, self.seq_len, dropout)
        else:
            # Minimal embedding fallback for the *w/o STCE* ablation.
            self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # --- GraphMamba UNet ---
        self.encoder_stages = nn.ModuleList(
            [self._build_stage() for _ in range(self.depth)]
        )
        down = max(0, self.depth - 1)
        self.downsamples = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, 3, stride=2, padding=1) for _ in range(down)]
        )
        self.upsamples = nn.ModuleList(
            [nn.ConvTranspose1d(d_model, d_model, 4, stride=2, padding=1)
             for _ in range(down)]
        )
        self.decoder_stages = nn.ModuleList(
            [self._build_stage() for _ in range(down)]
        )
        self.skip_projs = nn.ModuleList(
            [nn.Linear(d_model * 2, d_model) for _ in range(down)]
        )
        self.bottleneck_stage = self._build_stage()

        self.out_norm = RMSNorm(d_model)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)

        if self.uq_mode:
            # --- uncertainty-aware heads (PHQC_Engine, --cqr) ---
            # Node-based quantile head -> 3 channels per feature (mid, Δlo, Δhi).
            self.quant_head = nn.Linear(d_model, 3 * self.F)
            # Distribution-based Gaussian head -> (mu, logvar) per feature.
            self.dist_head = nn.Linear(d_model, 2 * self.F)
            # Parameter-based: extra dropout on features before the mean head.
            self.mc_dropout = nn.Dropout(dropout)
        else:
            # --- point head (BaseEngine, default) ---
            self.point_head = nn.Linear(d_model, self.F)

    def _build_stage(self):
        return nn.ModuleList([
            GMambaBlock(
                self.d_model, self.d_state, self.d_conv, self.expand,
                self.dropout_p, use_graph=self.use_gmamba, emb_dim=self.emb_dim,
            )
            for _ in range(self.num_layers)
        ])

    def _run_stage(self, stage, h):
        for block in stage:
            h = block(h, self.adj)
        return h

    def _resample(self, h, layer):  # h: (B, N, T, D)
        B, N, T, D = h.shape
        x = h.reshape(B * N, T, D).transpose(1, 2)
        x = layer(x)
        return x.transpose(1, 2).reshape(B, N, -1, D)

    @staticmethod
    def _match_length(t, target_len):
        cur = t.size(2)
        if cur == target_len:
            return t
        if cur > target_len:
            return t[:, :, :target_len, :]
        return F.pad(t, (0, 0, 0, target_len - cur))

    # -- backbone -----------------------------------------------------------

    def _backbone(self, x):  # x: (B, T, N, F) -> Z: (B, N, H, D)
        # (B, T, N, F) -> (B, N, T, F)
        x = x.permute(0, 2, 1, 3).contiguous()

        if self.use_stce:
            h = self.stce(x, self.adj)                    # (B, N, T, D)
        else:
            h = self.input_proj(x)
        h = h + self.pos_emb

        # Encoder + skips.
        skips = []
        for idx, stage in enumerate(self.encoder_stages):
            h = self._run_stage(stage, h)
            if idx < len(self.downsamples):
                skips.append(h)
                h = self._resample(h, self.downsamples[idx])

        # Bottleneck.
        h = self._run_stage(self.bottleneck_stage, h)

        # Decoder with skip fusion.
        for stage, up, proj in zip(
            reversed(self.decoder_stages),
            reversed(self.upsamples),
            reversed(self.skip_projs),
        ):
            h = self._resample(h, up)
            skip = skips.pop()
            h = self._match_length(h, skip.size(2))
            h = proj(torch.cat([h, skip], dim=-1))
            h = self._run_stage(stage, h)

        h = self._match_length(h, self.seq_len)
        h = self.out_norm(h)                              # (B, N, T, D)

        # Project time T -> horizon.
        h = h.permute(0, 1, 3, 2)                         # (B, N, D, T)
        h = self.time_proj(h).permute(0, 1, 3, 2)         # (B, N, H, D)
        return h

    # -- forward ------------------------------------------------------------

    def forward(self, x):  # x: (B, T, N, F)
        z = self._backbone(x)                             # (B, N, H, D)
        B, N, H, D = z.shape

        # Point-prediction mode (default, BaseEngine): a single regression head
        # returning (B, H, N, F) — the same contract as EnergyMamba/TrustEnergy.
        if not self.uq_mode:
            out = self.point_head(z)                      # (B, N, H, F)
            return out.permute(0, 2, 1, 3).contiguous()   # (B, H, N, F)

        # Node-based quantile head -> ordered (lo, mid, hi).
        q = self.quant_head(z).reshape(B, N, H, self.F, 3)
        mid = q[..., 0]
        lo = mid - F.softplus(q[..., 1])
        hi = mid + F.softplus(q[..., 2])

        # Distribution-based Gaussian head -> (mu, logvar).
        if self.use_dist:
            d = self.dist_head(z).reshape(B, N, H, self.F, 2)
            mu = d[..., 0]
            logvar = d[..., 1]
        else:
            mu = mid
            logvar = torch.zeros_like(mid)

        # Parameter-based consistency: variance of the mean head under two
        # dropout masks (non-trivial only while dropout is active / training).
        if self.use_param and self.training:
            m1 = self.dist_head(self.mc_dropout(z)).reshape(B, N, H, self.F, 2)[..., 0]
            m2 = self.dist_head(self.mc_dropout(z)).reshape(B, N, H, self.F, 2)[..., 0]
            mc_var = 0.5 * (m1 - m2) ** 2
        else:
            mc_var = torch.zeros_like(mid)

        # Reshape every head to (B, H, N, F) for the engine.
        def to_bhnf(t):
            return t.permute(0, 2, 1, 3).contiguous()

        return {
            "q_lo": to_bhnf(lo),
            "q_mid": to_bhnf(mid),
            "q_hi": to_bhnf(hi),
            "mu": to_bhnf(mu),
            "logvar": to_bhnf(logvar),
            "mc_var": to_bhnf(mc_var),
        }
