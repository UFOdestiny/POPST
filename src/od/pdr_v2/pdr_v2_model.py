"""PDR_v2 model.

The forecasting architecture intentionally stays identical to PDR.  PDR_v2
only changes the training objective in :mod:`pdr_v2_engine`.
"""

from src.od.pdr.pdr_model import PDR


class PDR_v2(PDR):
    """PDR with the PDR_v2 name; see :class:`PDR` for the architecture."""

    pass
