"""PDR ablation without origin/destination zone embeddings."""

from src.od.pdr.pdr_model import PDR


class PDR_no_zone_embed(PDR):
    """Remove learned zone identities while retaining all demand contexts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_embed = None
        self.dest_embed = None
        self.origin_proj = None
        self.dest_proj = None

    def _encode(self, x):
        bsz, num_origins, num_destinations, _ = x.shape
        pair = self.pair_stem(x)
        origin_context = self.origin_stem(x.sum(dim=2))
        dest_context = self.dest_stem(x.sum(dim=1))
        global_context = self.global_stem(x.sum(dim=(1, 2)))

        h = (
            pair
            + origin_context.unsqueeze(2)
            + dest_context.unsqueeze(1)
            + global_context.view(bsz, 1, 1, self.context_dim)
        )
        for layer in self.spatial_layers:
            h = layer(h, self.A_q, self.A_h)

        h = h.unsqueeze(3).expand(
            bsz, num_origins, num_destinations, self.horizon, self.context_dim
        )
        return self.context_norm(h)
