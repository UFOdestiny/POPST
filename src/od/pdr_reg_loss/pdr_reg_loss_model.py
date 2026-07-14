"""PDR regression model with an OD-marginal-regularized loss."""

from src.od.pdr_reg.pdr_reg_model import PDRReg


class PDRRegLoss(PDRReg):
    """The architecture is identical to :class:`PDRReg`.

    ``pdr_reg_loss`` differs only in the registered training loss selected by
    its runner; keeping a named subclass makes checkpoints and logs explicit.
    """

    pass
