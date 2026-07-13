"""PDR ablation without OD spatial message passing."""

from src.od.pdr.pdr_model import PDR


class PDR_no_spatial(PDR):
    """Remove every ODSeparableSpatialBlock from the PDR encoder."""

    def __init__(self, *args, **kwargs):
        kwargs["num_spatial_layers"] = 0
        super().__init__(*args, **kwargs)
