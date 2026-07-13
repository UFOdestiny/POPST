"""PDR ablation without aggregate origin/destination/global contexts."""

import torch

from src.od.pdr.pdr_model import PDR


class PDR_no_context(PDR):
    """Use pair histories only, removing all three aggregate temporal stems."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_stem = None
        self.dest_stem = None
        self.global_stem = None

    def _encode(self, x):
        bsz, num_origins, num_destinations, _ = x.shape
        pair = self.pair_stem(x)

        origin_ids = torch.arange(num_origins, device=x.device, dtype=torch.long)
        dest_ids = torch.arange(num_destinations, device=x.device, dtype=torch.long)
        origin_embed = self.origin_proj(self.origin_embed(origin_ids).to(pair.dtype))
        dest_embed = self.dest_proj(self.dest_embed(dest_ids).to(pair.dtype))

        h = (
            pair
            + origin_embed.view(1, num_origins, 1, self.context_dim)
            + dest_embed.view(1, 1, num_destinations, self.context_dim)
        )
        for layer in self.spatial_layers:
            h = layer(h, self.A_q, self.A_h)

        h = h.unsqueeze(3).expand(
            bsz, num_origins, num_destinations, self.horizon, self.context_dim
        )
        return self.context_norm(h)
