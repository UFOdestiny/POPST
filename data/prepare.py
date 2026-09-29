"""Array preparation API; paths and fixed settings are supplied by notebooks."""

from dataclasses import dataclass
from pathlib import Path
from data.preprocessing import generate


@dataclass(frozen=True)
class Preparation:
    data_path: str
    dataset: str
    years: str
    output_root: str
    fmt: str = "NDT"
    seq_length_x: int = 12
    seq_length_y: int = 1
    clip_neg: bool = False
    per_channel: bool = False
    log1p: bool = False


def prepare_array(
    data_path,
    dataset,
    years,
    output_root,
    fmt="NDT",
    seq_length_x=12,
    seq_length_y=1,
    clip_neg=False,
    per_channel=False,
    log1p=False,
):
    """Generate windows and train-fitted scaling in an explicit local directory."""
    cfg = Preparation(
        str(data_path),
        dataset,
        years,
        str(output_root),
        fmt,
        seq_length_x,
        seq_length_y,
        clip_neg,
        per_channel,
        log1p,
    )
    generate(cfg)
    return Path(output_root) / dataset / years
