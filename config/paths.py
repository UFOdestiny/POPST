"""Deployment paths. Relative values are resolved against the repository, never cwd."""

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path
    source: Path
    datasets: Path
    results: Path
    cache: Path
    models: Path

    def resolve(self, value, base=None):
        p = Path(value).expanduser()
        return p if p.is_absolute() else (base or self.root) / p

    def ensure_dataset_link(self):
        if self.datasets.is_symlink():
            if self.datasets.resolve() != self.source.resolve():
                raise ValueError(
                    f"Dataset link points to {self.datasets.resolve()}, expected {self.source}"
                )
        elif not self.datasets.exists():
            if not self.source.is_dir():
                raise FileNotFoundError(self.source)
            self.datasets.parent.mkdir(parents=True, exist_ok=True)
            self.datasets.symlink_to(self.source, target_is_directory=True)


def load_paths(env_file=None):
    env_file = Path(env_file or os.environ.get("POPST_ENV_FILE", ROOT / ".env")).resolve()
    values = {**dotenv_values(env_file), **os.environ}

    def path(key, default):
        value = values.get(key, default)
        p = Path(value).expanduser()
        return Path(os.path.abspath(ROOT / p)) if not p.is_absolute() else p

    return Paths(
        ROOT,
        path("POPST_DATA_SOURCE", "./datasets"),
        path("POPST_DATA_ROOT", "./datasets"),
        path("POPST_RUN_ROOT", "./results"),
        path("POPST_CACHE_ROOT", "./artifacts/cache"),
        path("POPST_MODEL_ROOT", "./artifacts/models"),
    )
