# POPST

POPST (Probabilistic and OD Prediction for SpatioTemporal data) is a unified benchmarking framework for spatiotemporal forecasting. It covers both **node-level flow prediction** and **origin-destination (OD) matrix forecasting**, with a shared training engine, logging system, evaluation pipeline, and Conformal Quantile Regression (CQR) wrapper.

## Key Features

- **Shared Runner**: `run_experiment()` in `base/runner.py` — every model's `main.py` reduces to a config dict + model class + one function call.
- **Unified Engine**: `BaseEngine` handles training loops, early stopping, checkpointing, and evaluation.
- **Conformal Quantile Regression**: `CQR_Engine` wraps any point-prediction model to produce calibrated prediction intervals with coverage guarantees.
- **Comprehensive Metrics**: MAE, RMSE, MAPE, KL divergence, CRPS, MPIW, WINK, Coverage, Interval Score, quantile loss.
- **MinMaxScaler**: Single, simple normalization to [0, 1] with `info.json` serialization for reproducible inverse transforms.
- **Config-driven Dataset Registry**: `utils/registry.yaml` defines all dataset paths and metadata — no hardcoded dicts.
- **PyTorch DataLoader**: `TimeSeriesDataset` + `LoaderAdapter` wraps standard `torch.utils.data.DataLoader` with `num_workers` and `pin_memory` support.
- **Efficiency Profiling**: Automatic hardware info, memory usage, inference time, and FLOPs reporting after every run.
- **Auto-Configuration**: `seq_len`, `horizon`, `input_dim`, `output_dim` are auto-filled from `info.json` — no manual specification needed.
- **Cross-Platform**: Runs on both Linux (HPC cluster) and Windows with automatic path and resource detection.

## Repository Layout

```
POPST/
  base/                   Core abstractions
    runner.py             Shared experiment runner (run_experiment)
    model.py              BaseModel, QuantileOutputLayer
    engine.py             BaseEngine (training, evaluation, checkpointing)
    CQR_engine.py         Conformal Quantile Regression engine
    metrics.py            All metric functions and the Metrics tracker
    efficiency.py         Hardware info, memory/FLOPs/inference profiling
  utils/                  Shared utilities
    args.py               Argument parser, path config, set_seed, check_quantile
    dataloader.py         TimeSeriesDataset, LoaderAdapter, dataset registry (YAML)
    generate.py           Data generation and MinMaxScaler
    registry.yaml         Dataset metadata (paths, adjacency, node counts)
    graph_algo.py         Graph normalization (symmetric, asymmetric, Chebyshev, scaled Laplacian)
    get_adj_mat.py        Adjacency matrix construction from geographic data
    log.py                Logger with rotating file handler
  src/
    flow/                 Flow prediction models
    od/                   OD matrix prediction models
  jobs/                   Slurm job scripts for HPC
  res.py                  Result visualization and analysis
```

## Supported Models

### Flow Prediction

Located in `src/flow/`:

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNET, STGCN, STGODE, UQGNN |
| **Sequence Models** | LSTM, Transformer, PatchTST, Mamba (variants 1-7) |
| **LLM-Based** | STLLM, STLLM2 |
| **Probabilistic** | GluonTS |
| **Baselines** | HL (Historical Last), STTN |

### OD Matrix Prediction

Located in `src/od/`:

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
- PyTorch >= 1.10

```bash
pip install -r requirements.txt
```

### Data Preparation

Datasets follow this directory structure:

```
<data_root>/
  <dataset_name>/
    adj.npy                 # Adjacency matrix (N x N)
    <year>/
      his.npz               # Normalized data + scaler params (min, max)
      info.json             # Data metadata (shape, scaler, splits, config)
      idx_train.npy          # Training sample indices
      idx_val.npy            # Validation sample indices
      idx_test.npy           # Test sample indices
```

Use `utils/generate.py` to create `his.npz`, `info.json`, and index files from raw data:

```bash
# Flow data
python utils/generate.py --dataset Tally_User --years 2018 --mode flow

# OD data
python utils/generate.py --dataset nyc_bike --years 2018 --mode od
```

Use `utils/get_adj_mat.py` to build adjacency matrices from geographic shapefiles.

Register new datasets in `utils/registry.yaml`:

```yaml
MyDataset:
  data: /MyDataset
  adj: /MyDataset/adj.npy
  nodes: 100
```

### Usage

Each model's `main.py` uses the shared runner:

```bash
# Flow prediction
python src/flow/stgcn/main.py --dataset NYISO --years 2018

# OD prediction
python src/od/gwnet/main.py --dataset nyc_taxi_od --years 2018

# With Conformal Quantile Regression
python src/flow/gwnet/main.py --dataset NYISO --years 2018 --quantile --quantile_alpha 0.1

# Test mode
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test

# Export predictions
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test --export
```

### HPC (Slurm)

Submit experiment jobs from the `jobs/` directory:

```bash
sbatch jobs/half.sh        # Half experiment (all flow models)
bash jobs/run_all.sh       # Submit all experiments
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

### Quantile Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--quantile` | `False` | Enable CQR prediction intervals |
| `--quantile_alpha` | `0.1` | Significance level for prediction intervals (1 - coverage) |

### System Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--device` | `cuda` | Device to use (`cuda` or `cpu`) |
| `--mode` | `train` | Execution mode: `train` or `test` |
| `--model_path` | -- | Path to a specific model checkpoint (for test mode) |
| `--export` | `False` | Save predictions and test data as `.npy` files |
| `--proj` | -- | Project name for organizing results into sub-folders |
| `--comment` | -- | Optional comment for the experiment |
| `--not_print_args` | `False` | Suppress argument logging at startup |

### Auto-Configuration

When `seq_len`, `horizon`, `input_dim`, or `output_dim` are not specified via CLI, they are automatically filled from the dataset's `info.json`:

- **seq_len** from `config.seq_length_x` (fallback: 12)
- **horizon** from `config.seq_length_y` (fallback: 1)
- **input_dim** from last dimension of `raw_data.shape` (fallback: 1)
- **output_dim** defaults to `input_dim` (models predict all input features)

For OD models that reshape input dimensions, `setup()` can override these values after auto-fill.

## Architecture

### Execution Flow

1. **Parse arguments**: CLI args + model-specific args merged
2. **Auto-fill**: `seq_len`, `horizon`, `input_dim`, `output_dim` from `info.json`
3. **Load data**: `TimeSeriesDataset` -> `DataLoader` -> `LoaderAdapter`
4. **Setup callback**: Model-specific preprocessing (adjacency matrices, etc.)
5. **Print args**: Grouped display (Data / Model / Training / System / Model-specific)
6. **Build model**: Construct model with final arg values
7. **Create engine**: `BaseEngine` or `CQR_Engine` with optimizer and scheduler
8. **Train**: Epoch loop with early stopping, validation, periodic test evaluation
9. **Test**: Per-horizon metric evaluation with optional export
10. **Profile**: Hardware info, memory, inference time, FLOPs

### Key Design Decisions

- **run_experiment()** orchestrates the full pipeline. Each model only provides: `add_args()`, `build_model()`, and optionally `setup()`, `make_optimizer()`, `make_scheduler()`.
- **BaseModel** provides `param_num()`, `horizon` property, and dimension tracking.
- **BaseEngine** handles gradient clipping, LR scheduling, early stopping, checkpointing, and per-horizon test evaluation.
- **CQR_Engine** extends any engine with conformal calibration for valid coverage prediction intervals.
- **profile_efficiency** runs after every experiment to report hardware, memory, timing, and FLOPs.

## Result Analysis

```bash
# Compare all models on a result directory (auto-discovers models/datasets)
python res.py --path result/Test

# Filter to specific datasets or models
python res.py --path result/Test --datasets nyc_mobility --models STGCN GWNET

# Detailed summary of a single log file
python res.py --log result/Test/STGCN/nyc_mobility/2026-04-16_09-08-38.log

# Select best log by RMSE instead of MAE
python res.py --path result/Test --select RMSE
```

## Acknowledgements

This repository re-implements open-source spatiotemporal forecasting baselines. Please cite the original authors when using specific models.
