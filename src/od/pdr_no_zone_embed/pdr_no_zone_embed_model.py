"""PDR ablation without origin/destination zone embeddings."""

from src.od.pdr.pdr_model import PDR


class PDR_no_zone_embed(PDR):
    """Ablate learned origin/destination identities, retaining demand context."""

    def __init__(self, *args, **kwargs):
        kwargs["use_zone_embeddings"] = False
        super().__init__(*args, **kwargs)
