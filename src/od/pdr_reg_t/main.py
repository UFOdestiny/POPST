import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from pdr_reg_t_engine import PDRRegStudentTEngine
from pdr_reg_t_model import PDRRegStudentT
from src.od.pdr_reg_dist_runner import run_pdr_reg_distribution


if __name__ == "__main__":
    run_pdr_reg_distribution(
        "PDR_REG_T",
        PDRRegStudentT,
        PDRRegStudentTEngine,
        student_t=True,
    )
