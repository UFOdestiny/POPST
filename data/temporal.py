"""Calendar periods and training-only history shared by OD baselines."""

from pathlib import Path
import numpy as np
import pandas as pd


def steps_per_day(frequency):
    step = pd.Timedelta(frequency)
    day = pd.Timedelta(days=1)
    if step <= pd.Timedelta(0) or day % step:
        raise ValueError("frequency must divide one day exactly")
    return int(day / step)


def training_end(folder, horizon):
    """Exclusive end of all training targets, never inferred from total length."""
    indices = np.load(Path(folder) / "idx_train.npy")
    return int(indices.max()) + horizon + 1
