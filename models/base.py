import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    # Whether the model can serve as a CQR quantile regressor — i.e. its final
    # layer is sized by ``output_dim`` so the runner can widen it to 3*F to emit
    # (q_lo, q_mid, q_hi) per feature.  Set False for models whose output width
    # is structurally locked to input_dim (autoregressive models that feed the
    # prediction back into the input) or that emit their own distribution.
    cqr_compatible = True

    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])


class BaseODModel(BaseModel):
    """OD forecasting with independent mobility channels sharing one backbone.

    Input (B,T,N,N,D) is folded to (B*D,T,N,N) for forward_single, then
    unfolded to (B,H,N,N,D). Destinations are features: input_dim and
    output_dim both equal node_num. A 4-D input denotes one channel.
    This is an OD adaptation, not a claim of source-paper equivalence.
    Joint-channel or distributional models may override forward."""

    # OD models emit a plain (B, H, N, N, D) tensor and are not the CQR
    # quantile regressor by default (the conformal path is feature-wise and
    # designed for flow models); keep them off the calibration.mode gate unless a
    # subclass opts in.
    cqr_compatible = False

    def forward(self, x, label=None):
        """Channel-as-batch wrapper around :meth:`forward_single`.

        ``x`` is ``(B, T, N, N, D)``; returns ``(B, horizon, N, N, D)``.
        Accepts a 4-D ``(B, T, N, N)`` tensor too (single channel, D=1) so the
        efficiency profiler and any single-channel caller still work.
        """
        x, b, d, squeeze_back = self._fold_channels(x)
        out = self.forward_single(x, label=label)  # (B*D, H, N, N)
        return self._unfold_channels(out, b, d, squeeze_back)

    # -- channel-as-batch helpers (reusable by tuple-returning subclasses) ---

    def _fold_channels(self, x):
        """Fold the D mobility channels into the batch dim.

        ``(B, T, N, N, D)`` -> ``(B*D, T, N, N)``.  Returns
        ``(x_folded, B, D, squeeze_back)``; ``squeeze_back`` is True when the
        input was 4-D (single channel) and the output should drop its channel
        axis again.
        """
        squeeze_back = False
        if x.dim() == 4:  # (B, T, N, N) — treat as a single channel
            x = x.unsqueeze(-1)
            squeeze_back = True
        b, t, n, m, d = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(b * d, t, n, m)
        return x, b, d, squeeze_back

    def _unfold_channels(self, out, b, d, squeeze_back=False):
        """Inverse of :meth:`_fold_channels` for a model output.

        ``(B*D, H, N, N)`` -> ``(B, H, N, N, D)`` (or ``(B, H, N, N)`` when
        ``squeeze_back``)."""
        _, h, n, m = out.shape
        out = out.reshape(b, d, h, n, m).permute(0, 2, 3, 4, 1)
        if squeeze_back:
            out = out.squeeze(-1)
        return out

    @abstractmethod
    def forward_single(self, x, label=None):
        """Run the single-channel backbone.

        ``x`` is ``(B', T, N, N)`` where ``B' = B·D`` (channels folded into the
        batch).  Must return ``(B', horizon, N, N)`` — i.e. the legacy
        single-channel OD output.  Build layers as in the single-channel model
        (``input_dim = output_dim = node_num``).

        Distribution models (such as STZINB) that emit a *tuple* of parameter
        tensors instead override :meth:`forward` directly and reuse
        :meth:`_fold_channels` / :meth:`_unfold_channels`.
        """
        raise NotImplementedError
