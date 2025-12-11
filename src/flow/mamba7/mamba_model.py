import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class AdaptiveGraphLearner(nn.Module):
    """Lightweight adaptive graph learning module."""

    def __init__(self, node_num, embed_dim=16, base_adj=None):
        super().__init__()
        self.node_num = node_num
        self.node_vec1 = nn.Parameter(torch.randn(node_num, embed_dim))
        self.node_vec2 = nn.Parameter(torch.randn(node_num, embed_dim))
        if base_adj is not None:
            self.register_buffer("base_adj", base_adj.clone().detach().float())
        else:
            self.base_adj = None

    def forward(self):
        adaptive_adj = torch.matmul(self.node_vec1, self.node_vec2.transpose(0, 1))
        adaptive_adj = F.relu(adaptive_adj)
        adaptive_adj = F.softmax(adaptive_adj, dim=-1)
        if self.base_adj is not None:
            adj = adaptive_adj + self.base_adj
        else:
            adj = adaptive_adj
        adj = adj + torch.eye(self.node_num, device=adj.device, dtype=adj.dtype)
        deg = adj.sum(dim=-1, keepdim=True)
        return adj / (deg + 1e-6)


class SimpleMambaBlock(nn.Module):
    """Residual Mamba block without explicit scale changes."""

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class UNetMamba(BaseModel):
    """
    UNet-style Mamba architecture for spatio-temporal prediction.

    Architecture:
    - Encoder: Multiple Mamba blocks with downsampling
    - Bottleneck: Mamba block at lowest resolution
    - Decoder: Multiple Mamba blocks with upsampling and skip connections
    - Adaptive graph learner before sequence modeling
    """

    def __init__(
        self, d_model, num_layers, feature, adj=None, graph_embed_dim=16, **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers  # Number of encoder/decoder levels
        self.feature = feature

        # Input projection: F → d_model
        self.input_proj = nn.Linear(self.feature, self.d_model)

        # Adaptive graph learner (simple structure-aware mixing)
        self.adaptive_graph = AdaptiveGraphLearner(
            node_num=self.node_num,
            embed_dim=graph_embed_dim,
            base_adj=adj,
        )

        # Lightweight stack of residual Mamba blocks
        self.blocks = nn.ModuleList(
            [SimpleMambaBlock(d_model=self.d_model) for _ in range(self.num_layers)]
        )

        # Output projection layers
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def apply_graph_structure(self, x):
        """Simple adaptive graph propagation with residual mixing."""
        adj = self.adaptive_graph()
        graph_x = torch.einsum("ij,btjd->btid", adj, x)
        return 0.5 * (x + graph_x)

    def forward(self, x):  # (B, T, N, F)
        B, T, N, F = x.shape

        # Input projection (per node/time) and graph-aware mixing
        x = self.input_proj(x)
        x = self.apply_graph_structure(x)

        # Merge batch and nodes → treat each node independently post graph mixing
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)

        # Store original for global skip
        x_input = x

        # Lightweight residual Mamba stack
        for block in self.blocks:
            x = block(x)

        # Match original sequence length if needed
        if x.shape[1] != T:
            x = x.permute(0, 2, 1)  # (B*N, d_model, T')
            x = nn.functional.interpolate(x, size=T, mode="linear", align_corners=False)
            x = x.permute(0, 2, 1)  # (B*N, T, d_model)

        # Global skip connection
        x = x + x_input

        # Project T → H
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.time_proj(x)  # (B*N, d_model, H)
        x = x.permute(0, 2, 1)  # (B*N, H, d_model)

        # Project feature back
        x = self.output_proj(x)  # (B*N, H, F)

        # Reshape back to (B, H, N, F)
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
