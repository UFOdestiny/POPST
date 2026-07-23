import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from pdr_reg_lap_engine import PDRRegLaplaceEngine
from pdr_reg_lap_model import PDRRegLaplace
from src.od.pdr_reg_dist_runner import run_pdr_reg_distribution


if __name__ == "__main__":
    run_pdr_reg_distribution(
        "PDR_REG_LAP", PDRRegLaplace, PDRRegLaplaceEngine
    )
