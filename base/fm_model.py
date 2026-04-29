import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time[:, None] * freqs[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class FlowMatchingWrapper(BaseModel):
    """Model-agnostic wrapper that adds flow-matching hooks to any flow model."""

    def __init__(
        self,
        base_model,
        context_dim=32,
        time_dim=32,
        flow_hidden_dim=64,
        node_emb_dim=8,
        dropout=0.1,
    ):
        self._validate_base_model(base_model)
        super().__init__(
            node_num=base_model.node_num,
            input_dim=base_model.input_dim,
            output_dim=base_model.output_dim,
            seq_len=base_model.seq_len,
            horizon=base_model.horizon,
        )
        self.base_model = base_model
        self.context_dim = context_dim
        self.time_dim = time_dim
        self.flow_hidden_dim = flow_hidden_dim
        self.node_emb_dim = node_emb_dim
        self.dropout = dropout

        self._base_forward_accepts_label = (
            len(inspect.signature(self.base_model.forward).parameters) > 1
        )

        self.history_encoder = nn.Sequential(
            nn.Linear(self.seq_len * self.input_dim, self.context_dim),
            nn.LayerNorm(self.context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if self.node_emb_dim > 0:
            self.node_emb = nn.Parameter(torch.randn(self.node_num, self.node_emb_dim))
        else:
            self.register_buffer("node_emb", torch.empty(self.node_num, 0))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
        )

        flow_input_dim = (
            self.output_dim  # noisy residual state x_t
            + self.output_dim  # point prediction condition
            + self.context_dim  # encoded input history
            + self.node_emb_dim  # node embeddings
            + self.time_dim  # continuous flow time embedding
        )
        self.flow_head = nn.Sequential(
            nn.Linear(flow_input_dim, self.flow_hidden_dim),
            nn.LayerNorm(self.flow_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.flow_hidden_dim, self.output_dim),
        )

    @staticmethod
    def _validate_base_model(base_model):
        required = ("node_num", "input_dim", "output_dim", "seq_len", "horizon")
        missing = [name for name in required if not hasattr(base_model, name)]
        if missing:
            raise TypeError(
                f"{type(base_model).__name__} cannot be wrapped for flow matching. "
                f"Missing required attribute(s): {', '.join(missing)}."
            )

    def _run_base_model(self, x, label=None):
        if self._base_forward_accepts_label:
            pred = self.base_model(x, label)
        else:
            pred = self.base_model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        return self._validate_base_prediction(pred)

    def _validate_base_prediction(self, pred):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"{type(self.base_model).__name__} must return a tensor prediction."
            )
        if pred.ndim != 4:
            raise ValueError(
                f"{type(self.base_model).__name__} must return a 4D tensor "
                f"(B, horizon, N, F), got shape {tuple(pred.shape)}."
            )
        expected = (self.horizon, self.node_num, self.output_dim)
        actual = tuple(pred.shape[1:])
        if actual != expected:
            raise ValueError(
                f"{type(self.base_model).__name__} must return shape "
                f"(B, {self.horizon}, {self.node_num}, {self.output_dim}), "
                f"got {tuple(pred.shape)}."
            )
        return pred

    def _encode_history(self, x):
        batch_size = x.shape[0]
        history = x.permute(0, 2, 1, 3).reshape(batch_size, self.node_num, -1)
        return self.history_encoder(history)

    def _expand_horizon(self, tensor):
        return tensor.unsqueeze(1).expand(-1, self.horizon, -1, -1)

    def _build_flow_context(self, x, point_pred):
        return {
            "point_condition": point_pred,
            "history_context": self._encode_history(x),
            "node_context": self.node_emb.unsqueeze(0).expand(point_pred.shape[0], -1, -1),
        }

    def _time_context(self, t):
        time_context = self.time_mlp(t).unsqueeze(1).expand(-1, self.node_num, -1)
        return self._expand_horizon(time_context)

    def forward_flow(self, x_t, flow_context, t):
        point_condition = flow_context["point_condition"].detach()
        history_context = self._expand_horizon(flow_context["history_context"])
        node_context = self._expand_horizon(flow_context["node_context"])
        time_context = self._time_context(t)

        flow_input = torch.cat(
            [x_t, point_condition, history_context, node_context, time_context], dim=-1
        )
        return self.flow_head(flow_input)

    def forward(self, x, label=None):
        return self._run_base_model(x, label=label)

    def forward_point(self, x, label=None):
        point_pred = self._run_base_model(x, label=label)
        return point_pred, self._build_flow_context(x, point_pred)
