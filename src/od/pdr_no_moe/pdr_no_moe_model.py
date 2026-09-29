"""PDR ablation replacing the mixture of experts with one MLP head."""

from src.od.pdr.pdr_model import PDR


class PDR_no_moe(PDR):
    """Ablate the regime router and all expert residual branches."""

    def __init__(self, *args, **kwargs):
        kwargs["use_moe"] = False
        super().__init__(*args, **kwargs)
