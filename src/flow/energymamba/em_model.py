"""EnergyMamba — Graph-Enhanced Selective State Space Model (GE-Mamba).

Reimplementation of the predictor described in EnergyMamba
(arXiv:2606.00506v1).  GE-Mamba pairs a graph-convolutional spatial
context extractor with a *spatial-conditioned* selective state-space
(Mamba) sequence model, arranged in a U-Net encoder/decoder over the
temporal axis.

Tensor convention inside the network: ``(B, N, T, D)`` — batch, node,
time, hidden — so the GCN can mix across ``N`` at every timestep while
the selective SSM scans along ``T`` per node.

Output width is driven by ``output_dim`` (a single ``output_proj``), so
the model is a drop-in CQR quantile regressor: in CQR mode the runner
widens ``output_dim`` to ``3*F`` and the engine reads the three channels
per feature as ``(q_mid, q_lo, q_hi)`` — see ``base/CQR_engine.py`` and
``src/flow/energymamba/ACQR_engine.py``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction)."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class GCN(nn.Module):
    """One-hop graph convolution applied independently at each timestep.

    ``Z_t = GELU( Ã_norm H_t W )`` with the pre-normalised adjacency
    ``Ã_norm = D̃^{-1/2}(A+I)D̃^{-1/2}`` supplied as a fixed ``(N, N)``
    tensor.
    """

    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, h, adj):  # h: (B, N, T, D), adj: (N, N)
        # mix across nodes: sum_m A[n,m] h[b,m,t,d]
        z = torch.einsum("nm,bmtd->bntd", adj, h)
        return F.gelu(self.lin(z))


class GEMambaSSM(nn.Module):
    """Selective state-space scan whose selectivity (Δ, B, C) is
    conditioned on the per-node spatial context produced by the GCN.

    Implemented on top of ``mamba_ssm``'s fused ``selective_scan_fn``:
    the input projection / causal conv follow the standard Mamba block,
    but the data-dependent ``Δ, B, C`` are regressed from the
    concatenation ``[x_conv ‖ z]`` so the spatial context steers the
    state-space recurrence (Eqs. 10-12 of the paper).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        # spatial-conditioned selectivity: project [x_conv ‖ z] -> Δ, B, C
        self.x_proj = nn.Linear(self.d_inner + d_model, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # state-space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, h, z):  # h, z: (B2, T, D)  with B2 = B * N
        B2, T, _ = h.shape

        xz = self.in_proj(h)                      # (B2, T, 2*d_inner)
        x, gate = xz.chunk(2, dim=-1)             # each (B2, T, d_inner)

        x = x.transpose(1, 2)                     # (B2, d_inner, T)
        x = self.conv1d(x)[..., :T]               # causal conv
        x = F.silu(x)

        # spatial-conditioned Δ, B, C from [x_conv ‖ z]
        x_t = x.transpose(1, 2)                   # (B2, T, d_inner)
        dbc = self.x_proj(torch.cat([x_t, z], dim=-1))
        dt, B_ssm, C_ssm = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj(dt).transpose(1, 2)  # (B2, d_inner, T)

        A = -torch.exp(self.A_log.float())        # (d_inner, d_state)
        B_ssm = B_ssm.transpose(1, 2).contiguous()  # (B2, d_state, T)
        C_ssm = C_ssm.transpose(1, 2).contiguous()

        y = selective_scan_fn(
            x.contiguous(),
            delta.contiguous(),
            A,
            B_ssm,
            C_ssm,
            self.D.float(),
            z=gate.transpose(1, 2).contiguous(),
            delta_bias=None,
            delta_softplus=True,
        )
        y = y.transpose(1, 2)                     # (B2, T, d_inner)
        return self.out_proj(y)


class GEMambaBlock(nn.Module):
    """Residual GE-Mamba block (Eqs. 13-15):

        Z   = GCN(RMSNorm(H), A)
        H   = H + Dropout( Mamba_→(RMSNorm(H), Z) + Mamba_←(flip) )
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gcn = GCN(d_model)
        self.fwd = GEMambaSSM(d_model, d_state, d_conv, expand)
        self.bwd = GEMambaSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, mamba, h, z):  # h, z: (B, N, T, D)
        B, N, T, D = h.shape
        h = h.reshape(B * N, T, D)
        z = z.reshape(B * N, T, D)
        out = mamba(h, z)
        return out.reshape(B, N, T, D)

    def forward(self, h, adj):  # h: (B, N, T, D)
        hn = self.norm(h)
        z = self.gcn(hn, adj)
        bip = self._scan(self.fwd, hn, z) + self._scan(
            self.bwd, hn.flip(2), z.flip(2)
        ).flip(2)
        return h + self.dropout(bip)


class EnergyMamba(BaseModel):
    """GE-Mamba U-Net predictor.

    Args (model-specific):
        d_model:    hidden dimension D.
        num_layers: GE-Mamba blocks per encoder/decoder stage (K).
        depth:      number of encoder stages (S); ``depth-1`` temporal
                    down/up-sampling levels around the bottleneck.
        d_state:    SSM state dimension.
        expand:     Mamba inner-dimension expansion factor.
        d_conv:     causal conv kernel width.
        dropout:    block dropout.
        adj:        pre-normalised dense adjacency, ``(N, N)`` array/tensor.
    """

    def __init__(
        self,
        d_model,
        num_layers,
        adj,
        depth=2,
        d_state=16,
        expand=2,
        d_conv=4,
        dropout=0.1,
        **args,
    ):
        super(EnergyMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = max(1, num_layers)
        self.depth = max(1, depth)
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.dropout_p = dropout

        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer("adj", adj)

        # input embedding F -> D, plus learnable temporal position embedding
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, self.seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # U-Net: encoder stages, bottleneck, symmetric decoder
        self.encoder_stages = nn.ModuleList(
            [self._build_stage() for _ in range(self.depth)]
        )
        down_blocks = max(0, self.depth - 1)
        self.downsamples = nn.ModuleList(
            [
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
                for _ in range(down_blocks)
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    self.d_model, self.d_model, kernel_size=4, stride=2, padding=1
                )
                for _ in range(down_blocks)
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [self._build_stage() for _ in range(down_blocks)]
        )
        self.skip_projs = nn.ModuleList(
            [nn.Linear(self.d_model * 2, self.d_model) for _ in range(down_blocks)]
        )
        self.bottleneck_stage = self._build_stage()

        self.out_norm = RMSNorm(self.d_model)
        # project time T -> horizon, and hidden D -> output_dim (3F under CQR)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.output_dim)

    def _build_stage(self):
        return nn.ModuleList(
            [
                GEMambaBlock(
                    self.d_model,
                    self.d_state,
                    self.d_conv,
                    self.expand,
                    self.dropout_p,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _run_stage(self, stage, h):  # h: (B, N, T, D)
        for block in stage:
            h = block(h, self.adj)
        return h

    # -- temporal resampling helpers (fold nodes into batch) ----------------

    def _resample(self, h, layer):  # h: (B, N, T, D)
        B, N, T, D = h.shape
        x = h.reshape(B * N, T, D).transpose(1, 2)  # (B*N, D, T)
        x = layer(x)
        return x.transpose(1, 2).reshape(B, N, -1, D)

    @staticmethod
    def _match_length(t, target_len):  # t: (B, N, T, D)
        cur = t.size(2)
        if cur == target_len:
            return t
        if cur > target_len:
            return t[:, :, :target_len, :]
        return F.pad(t, (0, 0, 0, target_len - cur))

    def forward(self, x):  # x: (B, T, N, F)
        B, T, N, Fdim = x.shape

        # (B, T, N, F) -> (B, N, T, D)
        h = self.input_proj(x).permute(0, 2, 1, 3).contiguous()
        h = h + self.pos_emb

        # encoder
        skips = []
        for idx, stage in enumerate(self.encoder_stages):
            h = self._run_stage(stage, h)
            if idx < len(self.downsamples):
                skips.append(h)
                h = self._resample(h, self.downsamples[idx])

        # bottleneck
        h = self._run_stage(self.bottleneck_stage, h)

        # decoder with skip connections
        for stage, upsample, proj in zip(
            reversed(self.decoder_stages),
            reversed(self.upsamples),
            reversed(self.skip_projs),
        ):
            h = self._resample(h, upsample)
            skip = skips.pop()
            target_len = skip.size(2)
            h = self._match_length(h, target_len)
            h = proj(torch.cat([h, skip], dim=-1))
            h = self._run_stage(stage, h)

        # restore the original sequence length for the time projection
        h = self._match_length(h, self.seq_len)
        h = self.out_norm(h)

        # project T -> horizon
        h = h.permute(0, 1, 3, 2)                 # (B, N, D, T)
        h = self.time_proj(h)                     # (B, N, D, H)
        h = h.permute(0, 1, 3, 2)                 # (B, N, H, D)

        # project D -> output_dim and reshape to (B, H, N, output_dim)
        out = self.output_proj(h)                 # (B, N, H, output_dim)
        return out.permute(0, 2, 1, 3).contiguous()
