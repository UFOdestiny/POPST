import torch
import torch.nn as nn

from base.model import BaseModel


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022), as used by the
    official PatchTST.  Normalizes each series over the time axis before
    patching and exactly reverses it after the head.  This is orthogonal to the
    dataset-level MinMax/log1p scaler the engine applies outside the model — it
    removes the per-window level/scale the global scaler cannot.
    """

    def __init__(self, num_features, eps=1e-5, affine=False, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(mode)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))  # time axis
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class PatchEmbedding(nn.Module):
    """Univariate patch embedding (channel-independent).

    Operates on a single series ``(B', L, 1)`` and projects each ``patch_len``
    window to ``d_model`` (official: ``W_P = Linear(patch_len, d_model)``).
    With ``padding_patch='end'`` the tail is replication-padded by ``stride``
    so no timesteps are dropped, adding one patch (matching the official).
    """

    def __init__(self, patch_len, stride, d_model, dropout=0.1, padding_patch="end"):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.pad = nn.ReplicationPad1d((0, stride)) if padding_patch == "end" else None

        self.projection = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B', L, 1)
        z = x.squeeze(-1)  # (B', L)
        if self.pad is not None:
            z = self.pad(z)  # (B', L + stride)
        # (B', num_patches, patch_len)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = self.projection(z)  # (B', num_patches, d_model)
        z = self.norm(z)
        return self.dropout(z)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class FlattenHead(nn.Module):
    """Official PatchTST head: flatten ``(num_patches, d_model)`` then a single
    linear to ``target_window`` (here ``horizon * series_out``)."""

    def __init__(self, d_model, num_patches, target_window, dropout=0.0):
        super(FlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(num_patches * d_model, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B', num_patches, d_model) -> (B', target_window)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(BaseModel):
    """PatchTST: A Time Series is Worth 64 Words (Nie et al., ICLR 2023).

    Reference: https://github.com/yuqinie98/PatchTST

    Faithful to the official design:
    1. **Channel independence** — every univariate series is processed
       independently with shared weights.  Our data is ``(B, T, N, F)`` (N
       nodes × F mobility channels), so both N and F are folded into the
       channel-independent batch axis ``(B*N*F, num_patches, patch_len)``.
    2. **RevIN** per-series instance normalization before patching / after head.
    3. **Patching** with end replication-padding.
    4. Transformer encoder over patches, single-linear flatten head.
    """

    def __init__(
        self,
        patch_len=2,
        stride=1,
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_layers=3,
        dropout=0.1,
        revin=True,
        affine=False,
        subtract_last=False,
        padding_patch="end",
        **args,
    ):
        super(PatchTST, self).__init__(**args)

        # auto-adjust patch_len / stride to fit seq_len
        if patch_len > self.seq_len:
            patch_len = max(1, self.seq_len // 2)
        if stride > patch_len:
            stride = max(1, patch_len // 2)

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_revin = revin

        # number of patches (account for end-padding)
        self.num_patches = max(1, (self.seq_len - patch_len) // stride + 1)
        if padding_patch == "end":
            self.num_patches += 1

        # Per-series output width: 1 in point mode (output_dim == input_dim == F),
        # or 3 under CQR where the runner widens output_dim to 3*F so each series
        # emits (q_lo, q_mid, q_hi).
        self.series_out = max(1, self.output_dim // self.input_dim)

        if self.use_revin:
            self.revin = RevIN(1, affine=affine, subtract_last=subtract_last)

        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout,
            padding_patch=padding_patch,
        )

        # learnable positional encoding over patches (official default: zeros init)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.pos_dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.head = FlattenHead(
            d_model=d_model,
            num_patches=self.num_patches,
            target_window=self.horizon * self.series_out,
            dropout=dropout,
        )

    def forward(self, x, label=None):
        # x: (B, seq_len, N, F)
        B, L, N, Fdim = x.shape

        # channel independence: fold BOTH node and feature into the batch axis
        # (B, L, N, F) -> (B, N, F, L) -> (B*N*F, L, 1)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N * Fdim, L, 1)

        if self.use_revin:
            x = self.revin(x, "norm")

        x = self.patch_embedding(x)  # (B*N*F, num_patches, d_model)
        x = self.pos_dropout(x + self.pos_embedding)
        x = self.encoder(x)  # (B*N*F, num_patches, d_model)
        x = self.head(x)  # (B*N*F, horizon * series_out)
        x = x.view(B * N * Fdim, self.horizon, self.series_out)

        if self.use_revin:
            x = self.revin(x, "denorm")

        # unfold back to (B, horizon, N, output_dim)
        # (B*N*F, horizon, series_out) -> (B, N, F, horizon, series_out)
        x = x.view(B, N, Fdim, self.horizon, self.series_out)
        # -> (B, horizon, N, F, series_out) -> (B, horizon, N, F*series_out)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = x.view(B, self.horizon, N, Fdim * self.series_out)
        return x
