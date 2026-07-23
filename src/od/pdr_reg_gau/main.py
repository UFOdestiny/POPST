import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from pdr_reg_gau_engine import PDRRegGaussianEngine
from pdr_reg_gau_model import PDRRegGaussian
from src.od.pdr_reg_dist_runner import run_pdr_reg_distribution


if __name__ == "__main__":
    run_pdr_reg_distribution(
        "PDR_REG_GAU", PDRRegGaussian, PDRRegGaussianEngine
    )
