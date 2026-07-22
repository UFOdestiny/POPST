import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from base.model import BaseODModel

def calculate_random_walk_matrix(adj_mx):
    """Returns the random walk adjacency matrix (for D_GCN)."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TemporalStem(nn.Module):
    """Summarizes a length-``T`` count history into a ``context_dim`` vector
    via a trend MLP over the full window plus hand-crafted local statistics
    (last value, first-order diff, range, mean), all in log1p space."""

    def __init__(self, input_horizon, context_dim, dropout=0.0):
        super().__init__()
        self.input_horizon = int(input_horizon)
        self.trend = MLP(input_horizon, context_dim, context_dim, dropout=dropout)
        self.local = MLP(4, context_dim, context_dim, dropout=dropout)
        self.mix = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.SiLU(),
            nn.LayerNorm(context_dim),
        )

    def forward(self, x):
        # x: [..., T]
        x_log = torch.log1p(torch.clamp(x, min=0.0))
        trend = self.trend(x_log)
        first = x_log[..., 0]
        last = x_log[..., -1]
        diff = x_log[..., -1] - x_log[..., -2] if self.input_horizon > 1 else torch.zeros_like(last)
        mean = x_log.mean(dim=-1)
        local = self.local(torch.stack([last, diff, last - first, mean], dim=-1))
        return self.mix(torch.cat([trend, local], dim=-1))


class ODSeparableSpatialBlock(nn.Module):
    """One diffusion-style message-passing step over the OD context, mixing
    each cell with its row (origin-conditioned, via ``A_q``) and column
    (destination-conditioned, via ``A_h``) neighborhoods."""

    def __init__(self, context_dim, dropout=0.0):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(context_dim * 3, context_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
        )
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, h, A_q, A_h):
        # h: [B, O, D, C]
        origin_msg = torch.einsum("ij,bjdc->bidc", A_q, h)
        dest_msg = torch.einsum("ij,bojc->boic", A_h, h)
        update = self.mix(torch.cat([h, origin_msg, dest_msg], dim=-1))
        return self.norm(h + update)


class ODRegimeExpertHead(nn.Module):
    """Mixture-of-experts residual head emitting the 3 raw ZINB logits
    (pre-activation ``n``, ``p``, ``pi``): a shared base plus a
    softmax-gated combination of ``num_experts`` regime-specific deltas."""

    def __init__(self, context_dim, hidden_dim=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.base = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.router = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, context):
        base = self.base(context)
        gate = torch.softmax(self.router(context), dim=-1)
        deltas = torch.stack([expert(context) for expert in self.experts], dim=-2)
        raw = base + (gate.unsqueeze(-1) * deltas).sum(dim=-2)
        return raw


class ODSingleZINBHead(nn.Module):
    """The shared (non-expert) ZINB head used for the no-MoE ablation.

    Keeping this head here makes the ablation a true architectural switch of
    :class:`PDR`, rather than a post-construction replacement of its head.
    """

    def __init__(self, context_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, context):
        return self.net(context)


class PDR(BaseODModel):
    """PDR-ZINB: a lightweight OD-aware encoder (Pair/origin/destination/global
    temporal stems + diffusion-style spatial mixing) with a mixture-of-experts
    residual head emitting zero-inflated negative-binomial parameters
    ``(n, p, pi)`` for sparse OD demand.

    Ported from the standalone ``ZINB/od_pdr_zinb_model.py`` prototype into the
    framework's channel-as-batch OD convention (see
    :class:`base.model.BaseODModel`): each mobility channel is folded into the
    batch dimension and run through the same single-channel encoder, exactly
    like :class:`src.od.stzinb.stzinb_model.STZINB`.  ``A_q``/``A_h`` (the two
    row/column-normalized random-walk matrices used for spatial mixing) are
    derived from the dataset adjacency the same way STZINB derives them.
    Trained by the ZINB negative log-likelihood (:func:`base.metrics.zinb_nll`)
    via :class:`base.engine.BaseEngine_OD_ZINB`; the point prediction is the
    ZINB mean ``E[y] = (1-pi)·n·(1-p)/p``.
    """

    cqr_compatible = False

    def __init__(
        self,
        A,
        node_num,
        input_dim,
        output_dim,
        seq_len,
        horizon,
        context_dim=64,
        zone_embed_dim=16,
        num_spatial_layers=2,
        num_experts=3,
        head_hidden_dim=128,
        dropout=0.0,
        use_aggregate_context=True,
        use_zone_embeddings=True,
        use_spatial_mixing=True,
        use_moe=True,
        **args,
    ):
        super(PDR, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)
        self.context_dim = int(context_dim)
        self.use_aggregate_context = bool(use_aggregate_context)
        self.use_zone_embeddings = bool(use_zone_embeddings)
        self.use_spatial_mixing = bool(use_spatial_mixing)
        self.use_moe = bool(use_moe)

        self.pair_stem = TemporalStem(seq_len, context_dim, dropout=dropout)
        if self.use_aggregate_context:
            self.origin_stem = TemporalStem(seq_len, context_dim, dropout=dropout)
            self.dest_stem = TemporalStem(seq_len, context_dim, dropout=dropout)
            self.global_stem = TemporalStem(seq_len, context_dim, dropout=dropout)

        if self.use_zone_embeddings:
            self.origin_embed = nn.Embedding(node_num, zone_embed_dim)
            self.dest_embed = nn.Embedding(node_num, zone_embed_dim)
            self.origin_proj = nn.Linear(zone_embed_dim, context_dim)
            self.dest_proj = nn.Linear(zone_embed_dim, context_dim)

        if self.use_spatial_mixing:
            self.spatial_layers = nn.ModuleList(
                [ODSeparableSpatialBlock(context_dim, dropout=dropout) for _ in range(int(num_spatial_layers))]
            )
        self.context_norm = nn.LayerNorm(context_dim)
        if self.use_moe:
            self.head = ODRegimeExpertHead(
                context_dim=context_dim,
                hidden_dim=head_hidden_dim,
                num_experts=num_experts,
                dropout=dropout,
            )
        else:
            self.head = ODSingleZINBHead(context_dim, head_hidden_dim, dropout)

        # Do not retain adjacency buffers when spatial mixing is ablated.  This
        # makes the no-spatial model independent of the graph at inference.
        if self.use_spatial_mixing:
            A_q = torch.from_numpy(calculate_random_walk_matrix(A).T.astype("float32"))
            A_h = torch.from_numpy(calculate_random_walk_matrix(A.T).T.astype("float32"))
            self.register_buffer("A_q", A_q)
            self.register_buffer("A_h", A_h)

    def _encode(self, x):
        # x: (B, O, D, T) -- a single mobility channel's OD history.
        bsz, num_origins, num_destinations, _ = x.shape
        pair = self.pair_stem(x)
        h = pair

        if self.use_aggregate_context:
            origin_context = self.origin_stem(x.sum(dim=2))
            dest_context = self.dest_stem(x.sum(dim=1))
            global_context = self.global_stem(x.sum(dim=(1, 2)))
            h = (
                h
                + origin_context.unsqueeze(2)
                + dest_context.unsqueeze(1)
                + global_context.view(bsz, 1, 1, self.context_dim)
            )

        if self.use_zone_embeddings:
            origin_ids = torch.arange(num_origins, device=x.device, dtype=torch.long)
            dest_ids = torch.arange(num_destinations, device=x.device, dtype=torch.long)
            origin_embed = self.origin_proj(self.origin_embed(origin_ids).to(dtype=pair.dtype))
            dest_embed = self.dest_proj(self.dest_embed(dest_ids).to(dtype=pair.dtype))
            h = (
                h
                + origin_embed.view(1, num_origins, 1, self.context_dim)
                + dest_embed.view(1, 1, num_destinations, self.context_dim)
            )

        if self.use_spatial_mixing:
            for layer in self.spatial_layers:
                h = layer(h, self.A_q, self.A_h)

        h = h.unsqueeze(3).expand(bsz, num_origins, num_destinations, self.horizon, self.context_dim)
        return self.context_norm(h)

    def forward(self, X, label=None):
        """``X`` is ``(B, T, N, N, D)``; returns ``(n, p, pi)``, each
        ``(B, horizon, N, N, D)``."""
        X, b, d, squeeze_back = self._fold_channels(X)  # (B*D, T, N, N)
        x = X.permute(0, 2, 3, 1)  # (B*D, O, D, T)

        context = self._encode(x)  # (B*D, O, Ddest, horizon, C)
        raw = self.head(context)
        # (B*D, O, Ddest, horizon) -> (B*D, horizon, O, Ddest) to match the
        # (batch, horizon, N, N) layout _unfold_channels expects.
        n = (F.softplus(raw[..., 0]) + 1e-4).permute(0, 3, 1, 2)
        p = torch.sigmoid(raw[..., 1]).clamp(1e-6, 1.0 - 1e-6).permute(0, 3, 1, 2)
        pi = torch.sigmoid(raw[..., 2]).clamp(1e-6, 1.0 - 1e-6).permute(0, 3, 1, 2)

        n = self._unfold_channels(n, b, d, squeeze_back)
        p = self._unfold_channels(p, b, d, squeeze_back)
        pi = self._unfold_channels(pi, b, d, squeeze_back)
        return n, p, pi
