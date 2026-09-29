# POPST

Version 2.0. See [release changes](CHANGELOG.md).

PyTorch models for node flow and origin–destination (OD) forecasting, with conformal
calibration. Model IDs and implementations are listed in `config/models/registry.yaml`.

## Run

Use the project Python environment from the repository root:

```bash
pip install --no-build-isolation -r requirements.txt
python run.py config/runs/dc_od_60min_bike/od_pdr_reg.yaml --dry-run
python run.py config/runs/dc_od_60min_bike/od_pdr_reg.yaml
python run.py config/suites/od_baselines.yaml
python run.py config/suites/od_zeropdr.yaml
```

Tested with Python 3.11 and Torch 2.10.0 + CUDA 12.8. Mamba builds against the active
Torch/CUDA installation. ST-LLM-plus needs local GPT-2 weights or downloads them on first use.

All supplied suites use seed **2026** and `devices: all`: one independent experiment per
visible GPU. Explicit lists such as `[cuda:0, cuda:2]` remain supported. A single model
uses one GPU. `all` requires a visible CUDA GPU and respects Slurm / `CUDA_VISIBLE_DEVICES`.
The baseline suite has **60 tasks**; the PDR suite has **54**. Neither launches calibration.

See the [configuration guide](config/README.md) for batch size and other overrides,
the [run catalog](config/runs/README.md) for calibration, and the
[OD implementation audit](models/od/README.md) for baseline provenance and limitations.

## Paths and outputs

Copy `.env.example` to `.env` on another machine. Environment variables override `.env`;
relative paths resolve from the repository root.

| Variable | Purpose / default |
|---|---|
| `POPST_DATA_SOURCE` | Shared dataset source; set for your installation |
| `POPST_DATA_ROOT` | Prepared datasets: `./datasets`, linked to the source |
| `POPST_RUN_ROOT` | Experiment outputs: `./results` |
| `POPST_CACHE_ROOT` | Hugging Face / Torch caches: `./artifacts/cache` |
| `POPST_MODEL_ROOT` | Local pretrained models: `./artifacts/models` |

Existing `HF_HOME` and `TORCH_HOME` settings take precedence. ST-LLM-plus checks
`<POPST_MODEL_ROOT>/<pretrained_model>` before using a Hugging Face model ID.
Shared datasets remain read only; generated data and results stay local.

Each run writes configurations, metadata, logs, status, and metrics to
`results/<experiment>/<run-id>/`. Neural training saves `best.pt` for evaluation;
optimizer-state training resumption is not supported. Enable `output.predictions`,
`output.test_inputs`, or `runtime.profile` for additional exports. Suite task logs and
status are under `results/suites/`; failures produce a nonzero exit after the batch finishes.

```python
from engine.reporting import report
report(root="results", output="results/summary.csv")
```

## Slurm

`run.sh` runs the OD baseline suite, then the PDR suite, with the configured Python
executable. Both receive the same command-line arguments; a baseline failure stops
the script before PDR starts. Its
`#SBATCH` directives request three GPUs, 12 CPUs, 64 GB RAM, and 23 hours. Edit those
directives or pass `sbatch` flags to change the allocation; `devices: all` uses only the
GPUs visible inside it. The default four threads per experiment fit three concurrent runs.

```bash
bash run.sh --dry-run     # Preview both suites; requires visible GPUs.
mkdir -p results
sbatch run.sh            # Allocate resources and execute.
```

`bash run.sh` runs on the current host without allocating resources. The batch log is
`results/st-<job-id>.out`.

## Data preparation

Fixed preprocessing settings and `RAW_ROOT`, `ASSET_ROOT`, and `PROCESSED_ROOT` live in
the original city notebooks. Download notebooks use `artifacts/raw/`; preparation
notebooks default to `datasets/raw/`. Set their `RAW_ROOT` to process a fresh download.
Generated arrays default to `artifacts/data/`; select them with `data.root: artifacts/data`.

```python
from engine.notebooks import execute_notebook
execute_notebook("notebooks/DC/DC_OD.ipynb", timeout=3600)
```

The executor saves a copy under `results/notebooks/`. New preparation fits scaling on
training observations and removes target overlap between multi-step splits. Existing
shared arrays need regeneration to use this protocol. `data/revision.py` provides the
separate chronological splits used by the review-driven experiment protocol.

## Development

`config/` holds experiment definitions; `models/` contains model implementations;
`engine/` handles execution and metrics; `data/` and `notebooks/` prepare datasets.
Generated `results/` and `artifacts/` are ignored by Git.

```bash
python -m pytest -q
ruff check run.py config data engine models tests --select F401,F841,F811,F821
bash -n run.sh
```

Tests exercise small fixtures, checkpoint replay, forecast causality, and suite scheduling.
Full benchmark training and the remaining uncertainty audit are still required for paper
comparisons; the repository does not establish SOTA performance. Notebook preparation
cells expose variables through `globals().update(result)`; validate them by execution
with the required raw data rather than applying Python-module unused-code fixes.
