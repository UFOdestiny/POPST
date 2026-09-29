# POPST

POPST (Probabilistic and OD Prediction for SpatioTemporal data) is a unified benchmarking framework for spatiotemporal forecasting. It covers both **node-level flow prediction** and **origin-destination (OD) matrix forecasting**, with shared runners, shared evaluation/metric infrastructure, and Conformal Quantile Regression (CQR).

## Key Features

- **Shared Runner**: `run_experiment()` in `base/runner.py` — every model's `main.py` reduces to a config dict + model class + one function call.
- **Unified Engines**: `BaseEngine` (point prediction) and a family of conformal/uncertainty engines (`CQR_Engine`, `ACQR_Engine`, `SCQR_Engine`, `PHQC_Engine`) all share the same runner, checkpointing, logging, and metric pipeline. The point vs. uncertainty engine is chosen by `--cqr`.
- **Conformalized Quantile Regression**: `CQR_Engine` (Romano et al., NeurIPS 2019) turns the model itself into the quantile regressor — the runner widens `output_dim` to `3*F` so the model emits `(q_lo, q_mid, q_hi)` per feature (trained with the pinball loss), then calibrates a conformal correction `Q` on the held-out validation split, yielding intervals with a finite-sample marginal coverage guarantee `>= 1 - alpha`. Enabled with `--cqr horizon` (one `Q` per forecast step) or `--cqr global` (single shared `Q`); `--cqr no` (default) runs the plain point model. Autoregressive models work too (they feed back only the median channel — see STGCN); distribution models (e.g. UQGNN) set `cqr_compatible = False` and are rejected with a clear error under `--cqr`.
- **Per-model uncertainty engines**: a model's `main.py` can register its own quantile engine via `engine_quantile_cls`, selected by the same `--cqr` switch while `--cqr no` still runs the point model on `BaseEngine`. `ACQR_Engine` (EnergyMamba) adds locally-adaptive, sequentially-updated conformal calibration; `SCQR_Engine` (TrustEnergy) adds a sequential conformal correction; `PHQC_Engine` (HealthMamba) trains three joint uncertainty heads and applies post-hoc quantile calibration with MC-dropout.
- **Comprehensive Metrics**: MAE, RMSE, MAPE, KL divergence, CRPS, MPIW, WINK, Coverage, Interval Score, quantile/pinball loss, multivariate-Gaussian NLL (MGAU), and zero-inflated negative-binomial NLL (ZINB).
- **MinMaxScaler**: Single normalization to [0, 1] with `info.json` serialization for reproducible inverse transforms.
- **Config-driven Dataset Registry**: `utils/registry.yaml` defines all dataset paths and metadata — no hardcoded dicts.  Node count is derived at runtime from `info.json`.
- **PyTorch DataLoader**: `TimeSeriesDataset` + `LoaderAdapter` wraps standard `torch.utils.data.DataLoader` with `num_workers` and `pin_memory` support.
- **Efficiency Profiling**: Automatic hardware info, memory usage, inference time, and FLOPs reporting after every run.
- **Auto-Configuration**: `seq_len`, `horizon`, `input_dim`, `output_dim` are auto-filled from `info.json` — no manual specification needed.
- **Cross-Platform**: Runs on both Linux (HPC cluster) and Windows with automatic path and resource detection.

## Repository Layout

```
POPST/
  base/                   Core abstractions
    runner.py             Shared experiment runner (run_experiment)
    model.py              BaseModel (forward) + BaseODModel (channel-as-batch OD)
    engine.py             BaseEngine + BaseEngine_OD / BaseEngine_OD_Stat
    CQR_engine.py         Conformalized Quantile Regression engine
    metrics.py            All metric functions and the Metrics tracker
    efficiency.py         Hardware info, memory/FLOPs/inference profiling
  utils/                  Shared utilities
    args.py               Shared CLI arguments, validation, path config, set_seed
    dataloader.py         TimeSeriesDataset, LoaderAdapter, dataset registry (YAML)
    generate.py           Data generation and MinMaxScaler
    registry.yaml         Dataset metadata (data paths, adjacency paths)
    graph_algo.py         Graph normalization (symmetric, asymmetric, Chebyshev, scaled Laplacian)
    get_adj_mat.py        Adjacency matrix construction from geographic data
    log.py                Logger with rotating file handler
    res.py                Result collection and analysis CLI
  src/
    flow/                 Flow prediction models (30 models)
      energymamba/        GE-Mamba model + ACQR_engine.py (Adaptive Sequential CQR)
      trustenergy/        MASTGNN model + SCQR_engine.py (Sequential CQR)
      healthmamba/        GraphMamba model + PHQC_engine.py (Post-Hoc Quantile Calibration)
    od/                   OD matrix prediction models (17 models)
  bash/                   Experiment scripts organized by region (CA, FL1, FL2, NY)
  jobs/                   Slurm job scripts for HPC
  requirements.txt        Python dependencies with pinned versions
```

## Supported Models

### Flow Prediction (`src/flow/`)

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNET, STGCN, STGODE |
| **Sequence Models** | LSTM, Transformer, PatchTST |
| **Mamba State-Space** | Mamba |
| **Uncertainty-Aware** | UQGNN, EnergyMamba (GE-Mamba + Adaptive Sequential CQR), TrustEnergy (MASTGNN + Sequential CQR), HealthMamba (GraphMamba + Post-Hoc Quantile Calibration) |
| **LLM-Based** | STLLM, STLLM2, STLLM3, STLLM4, STLLM5, STLLM6, STLLM7, STLLM8, STLLM9, ST-LLM-plus |
| **Probabilistic** | GluonTS |
| **Baselines** | HL (Historical Last), STTN |

### OD Matrix Prediction (`src/od/`)

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, GWNET, STGCN, STGODE |
| **Sequence / MLP** | LSTM, STTN, ODMixer |
| **Specialized OD** | STZINB, GMEL, HMDLF, MYOD |
| **Statistical** | ARIMA, SARIMA, VAR, HA (Historical Average) |
| **Baselines** | HL (Historical Last) |

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.8.0 with CUDA 12.8

```bash
# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### Data Preparation

Datasets follow this directory structure:

```
<data_root>/
  <dataset_name>/
    <adj_name>.npy          # Adjacency matrix (N x N)
    <year>/
      his.npz               # Normalized data + scaler params (min, max)
      info.json             # Data metadata (shape, scaler, splits, config)
      meta.json             # Scaler parameters and data shape
      idx_train.npy          # Training sample indices
      idx_val.npy            # Validation sample indices
      idx_test.npy           # Test sample indices
      idx_all.npy            # All sample indices
```

Use `utils/generate.py` to create `his.npz`, `info.json`, and index files from raw data:

```bash
# Flow data (default format: NDT = Time x Nodes x Features)
python utils/generate.py --data_path /path/to/raw.npy --dataset CAISO --years 2018 --fmt NDT

# OD data (format: NNDT = Origin x Destination x Features x Time) → stored as (T, N, N, D)
# D = mobility channels (e.g. taxi / fhv / bike); --per_channel --log1p suit sparse counts
python utils/generate.py --data_path /path/to/raw.npy --dataset nyc_taxi_od --years 2018 --fmt NNDT --per_channel --log1p
```

**Supported dimension formats**: `NDT` (Time×N×D), `NTD` (N×T×D), `NT` (N×T, auto-appends D=1), `NNDT` (OD matrix, D mobility channels)

Register new datasets in `utils/registry.yaml`:

```yaml
MyDataset:
  data: /MyDataset
  adj: /MyDataset/adj.npy
```

Node count is automatically read from `info.json` at runtime — no need to specify it in the registry.

Use `utils/get_adj_mat.py` to build adjacency matrices from geographic shapefiles.

### Usage

Each model's `main.py` uses the shared runner:

```bash
# Flow prediction
python src/flow/stgcn/main.py --dataset NYISO --years 2018

# OD prediction
python src/od/gwnet/main.py --dataset nyc_taxi_od --years 2018

# With Conformalized Quantile Regression
python src/flow/gwnet/main.py --dataset NYISO --years 2018 --cqr horizon --quantile_alpha 0.1

# Test mode
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test

# Export prediction archives
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test --export

# Organize results into a project folder
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --proj MyExperiment
```

### HPC (Slurm)

```bash
# Submit a general test job
sbatch jobs/test.sh

```

## Arguments Reference

### Data Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `CAISO` | Dataset name (must match `registry.yaml`) |
| `--years` | `2018` | Data sub-folder for the time period |
| `--seq_len` | auto | Input sequence length. Auto-filled from `info.json` if not specified |
| `--horizon` | auto | Prediction horizon. Auto-filled from `info.json` if not specified |
| `--input_dim` | auto | Number of input features. Auto-filled from data shape in `info.json` |
| `--output_dim` | auto | Number of output features. Defaults to `input_dim` if not specified |
| `--normalize` | `True` | Apply MinMaxScaler normalization. Use `--no_normalize` to disable |

### Training Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--bs` | `64` | Batch size |
| `--max_epochs` | `2000` | Maximum training epochs |
| `--patience` | `30` | Early stopping patience (epochs without validation improvement) |
| `--seed` | `2025` | Random seed for reproducibility |

### CQR Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--cqr` | `no` | Enable CQR and choose the conformal-correction granularity: `no` (disabled), `horizon` (one `Q` per forecast step), or `global` (single shared `Q`) |
| `--quantile_alpha` | `0.1` | Target miscoverage `alpha`; intervals target `1 - alpha` coverage, with quantile levels `alpha/2` and `1 - alpha/2` |

### System Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--device` | `cuda` | Device to use (`cuda` or `cpu`) |
| `--mode` | `train` | Execution mode: `train` or `test` |
| `--model_path` | -- | Path to a specific model checkpoint (for test mode) |
| `--export` | `False` | Save prediction/result archives alongside the final test evaluation |
| `--proj` | -- | Project name for organizing results into sub-folders |
| `--comment` | -- | Optional comment for the experiment |
| `--not_print_args` | `False` | Suppress argument logging at startup |

### Auto-Configuration

When `seq_len`, `horizon`, `input_dim`, or `output_dim` are not specified via CLI, they are automatically filled from the dataset's `info.json`:

- **seq_len** from `config.seq_length_x` (fallback: 12)
- **horizon** from `config.seq_length_y` (fallback: 1)
- **input_dim** from last dimension of `raw_data.shape` (fallback: 1)
- **output_dim** defaults to `input_dim` (models predict all input features)

For OD models (`od=True`), the runner overrides `input_dim`/`output_dim` to `node_num` after auto-fill (the N destinations are the feature axis; the D mobility channels are folded into the batch) — see *OD Models* below.

## Architecture

### Execution Flow

1. **Parse arguments**: CLI args + model-specific args merged
2. **Auto-fill**: `seq_len`, `horizon`, `input_dim`, `output_dim` from `info.json`
3. **Load data**: `TimeSeriesDataset` → `DataLoader` → `LoaderAdapter`
4. **Setup callback**: Model-specific preprocessing (adjacency matrices, etc.)
5. **Print args**: Grouped display (Data / Model / Training / Quantile / System / Model-specific)
6. **Build model**: Construct the requested backbone with final arg values
7. **Create engine**: `BaseEngine` (point) or the model's quantile engine (`CQR_Engine` / `ACQR_Engine` / `SCQR_Engine` / `PHQC_Engine`), selected by `--cqr`
8. **Train**: Epoch loop with early stopping and validation-based checkpoint selection
9. **Test / Export**: Final per-horizon evaluation with optional result export
10. **Profile**: Hardware info, memory, inference time, FLOPs

### Model Registration Pattern

Each model contributes only the pieces that are unique to it:

```python
def add_args(parser):
    """Add model-specific CLI arguments."""
    parser.add_argument("--hidden_dim", type=int, default=64)

def setup(args, data_path, adj_path, node_num, device, logger):
    """Model-specific preprocessing (adjacency matrices, etc.)."""
    adj = np.load(adj_path)
    return {"adj": torch.tensor(adj).to(device)}

def build_model(args, node_num, **ctx):
    """Construct the model using final arg values and setup context."""
    return MyModel(node_num, args.input_dim, args.output_dim, ctx["adj"])

if __name__ == "__main__":
    run_experiment(
        model_name="MyModel",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
    )
```

Optional callbacks include `make_optimizer`, `make_scheduler`, `load_data`, `engine_cls`, `engine_quantile_cls`, and `device_override`.

Statistical models (ARIMA, VAR, etc.) use `make_optimizer=NO_OPTIMIZER` to skip gradient-based training, with custom engine classes for their specific training loops.

### OD Models (`src/od/`)

OD datasets are `(T, N, N, D)` — every origin-destination pair carries **D mobility channels** (e.g. taxi / fhv / bike), so the dataloader yields 5-D batches `(B, T, N, N, D)`. Published OD models are single-channel: they treat the **N destinations as the feature axis** (`input_dim = output_dim = node_num`) and predict one `N×N` matrix.

`base.model.BaseODModel` reconciles the two with a **channel-as-batch** contract (one shared backbone, channels along the batch dim — the standard multimodal-baseline treatment): `forward()` folds the D channels into the batch, calls the model's `forward_single()` on the legacy 4-D `(B·D, T, N, N)` tensor, then unfolds back to `(B, horizon, N, N, D)`. A new OD model therefore just inherits `BaseODModel`, implements `forward_single`, and passes `od=True` to `run_experiment` — which centrally forces `input_dim = output_dim = node_num` and selects `BaseEngine_OD` (no manual `setup()` dimension overrides):

```python
class MyOD(BaseODModel):
    def forward_single(self, x, label=None):  # x: (B·D, T, N, N)
        ...                                    # return (B·D, horizon, N, N)

run_experiment(model_name="MyOD", add_args=add_args, build_model=build_model, od=True)
```

`BaseEngine_OD` reuses `BaseEngine` wholesale, overriding only the two tensor-shape seams (`_collect`, `_horizon_slice`) for the 5-D OD layout. Probabilistic OD models keep their distribution heads via a thin engine subclass: **STZINB** (zero-inflated negative binomial, `STZINB_Engine`) and **STTN** (multivariate Gaussian / `MGAU`, `STTN_Engine`). Statistical OD models (ARIMA, SARIMA, VAR, HA) share `BaseEngine_OD_Stat` and fit-and-forecast over plain `(T, N, N, D)` arrays.

### Graph Normalization

`utils/graph_algo.py` supports multiple adjacency matrix normalization methods:

| Method | Description | Used by |
| --- | --- | --- |
| `scalap` | Scaled (symmetric-normalized) Laplacian | STGCN, DSTAGNN |
| `normlap` | Normalized Laplacian | General |
| `symadj` | Symmetric adjacency | D2STGNN |
| `transition` | Random walk | General |
| `doubletransition` | Bidirectional random walk | GWNET, DCRNN, DGCRN, D2STGNN, STTN |
| `identity` | Identity matrix | Baselines |

> **ASTGCN** uses the *combinatorial* scaled Laplacian (`2(D − W)/λ_max − I`, computed locally in its `main.py`) rather than the shared `scalap` helper, matching the official `guoshnBJTU/ASTGCN-r-pytorch`.

## Result Analysis

```bash
# Compare all models on a result directory (auto-discovers models/datasets)
python utils/res.py --path result/MyExperiment

# Filter to specific datasets or models
python utils/res.py --path result/MyExperiment --datasets nyc_mobility --models STGCN GWNET

# Detailed summary of a single log file
python utils/res.py --log result/MyExperiment/STGCN/nyc_mobility/2026-04-16_09-08-38.log

# Select best log by RMSE instead of MAE
python utils/res.py --path result/MyExperiment --select RMSE
```

## Baseline Fidelity Notes

Flow baselines are ported to stay close to their official implementations while fitting the shared runner (input `(B, seq_len, N, F)`, output `(B, horizon, N, F)`, final projection sized by `output_dim` for CQR). A few conventions are worth knowing:

- **Multi-channel mobility data, no calendar features.** Flow datasets are `(T, N, 3)` mobility volumes (taxi / fhv|tnp / bike) with **no** time-of-day / day-of-week channels. Models whose official versions consume calendar features (DGCRN, D2STGNN) have those paths neutralized (fed zeros / a fixed embedding) rather than indexing mobility values as if they were time — set the model's time-feature flag only for datasets that actually append calendar channels.
- **CQR & autoregression.** Single-step autoregressive models (STGCN) feed back only the median channel and support `--cqr`; seq2seq decoders (DCRNN, DGCRN) feed the full prediction back as the next input and so set `cqr_compatible = False` (`--cqr` is rejected with a clear message). Run those without `--cqr`, or use CQR on a direct-output model.
- **Model-specific graph operators.** ASTGCN builds its own combinatorial scaled Laplacian; DSTAGNN uses a full-rank learnable spatial mask per Chebyshev order. See `src/flow/<model>/` and the graph-normalization table above.

Default hyperparameters track the original papers/configs where practical (e.g. DCRNN `n_filters=64`, AGCRN `cheb_k=3`); a few are scaled to the smaller graphs (`N≈69–77`). All flow baselines pass an end-to-end smoke run on `chicago_15min`.

## Acknowledgements

This repository re-implements open-source spatiotemporal forecasting baselines. Please cite the original authors when using specific models.
