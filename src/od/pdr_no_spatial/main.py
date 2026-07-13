import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from pdr_no_spatial_engine import PDR_no_spatial_Engine
from pdr_no_spatial_model import PDR_no_spatial
from src.od.pdr.ablation_runner import run_pdr_ablation


if __name__ == "__main__":
    run_pdr_ablation("PDR_no_spatial", PDR_no_spatial, PDR_no_spatial_Engine)
