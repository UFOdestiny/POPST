"""PDR ablation without aggregate origin/destination/global contexts."""

from src.od.pdr.pdr_model import PDR


class PDR_no_context(PDR):
    """Ablate origin, destination, and global aggregate temporal contexts."""

    def __init__(self, *args, **kwargs):
        kwargs["use_aggregate_context"] = False
        super().__init__(*args, **kwargs)
