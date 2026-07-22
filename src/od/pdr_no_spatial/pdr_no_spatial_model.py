"""PDR ablation without OD spatial message passing."""

from src.od.pdr.pdr_model import PDR


class PDR_no_spatial(PDR):
    """Ablate PDR's graph diffusion blocks and their adjacency inputs."""

    def __init__(self, *args, **kwargs):
        kwargs["use_spatial_mixing"] = False
        super().__init__(*args, **kwargs)
