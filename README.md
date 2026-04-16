# POPST

POPST (Probabilistic and OD Prediction for SpatioTemporal data) is a unified benchmarking framework for spatiotemporal forecasting. It covers both **node-level flow prediction** and **origin-destination (OD) matrix forecasting**, with a shared training engine, logging system, evaluation pipeline, and Conformal Quantile Regression (CQR) wrapper.

## Key Features

- **Shared Runner**: `run_experiment()` in `base/runner.py` — every model's `main.py` reduces to a config dict + model class + one function call.
- **Unified Engine**: `BaseEngine` handles training loops, early stopping, checkpointing, and evaluation.
- **Conformal Quantile Regression**: `CQR_Engine` wraps any point-prediction model to produce calibrated prediction intervals with coverage guarantees.
- **Comprehensive Metrics**: MAE, RMSE, MAPE, KL divergence, CRPS, MPIW, WINK, Coverage, Interval Score, quantile loss.
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
    model.py              BaseModel with forward(x, y) interface
    engine.py             BaseEngine (training, evaluation, checkpointing)
    CQR_engine.py         Conformal Quantile Regression engine
    metrics.py            All metric functions and the Metrics tracker
    efficiency.py         Hardware info, memory/FLOPs/inference profiling
  utils/                  Shared utilities
    args.py               Argument parser, path config, set_seed, check_quantile
    dataloader.py         TimeSeriesDataset, LoaderAdapter, dataset registry (YAML)
    generate.py           Data generation and MinMaxScaler
    registry.yaml         Dataset metadata (data paths, adjacency paths)
    graph_algo.py         Graph normalization (symmetric, asymmetric, Chebyshev, scaled Laplacian)
    get_adj_mat.py        Adjacency matrix construction from geographic data
    log.py                Logger with rotating file handler
    res.py                Result collection and analysis CLI
  src/
    flow/                 Flow prediction models (25 models)
    od/                   OD matrix prediction models (17 models)
  bash/                   Experiment scripts organized by region (CA, FL1, FL2, NY)
  jobs/                   Slurm job scripts for HPC
  requirements.txt        Python dependencies with pinned versions
```

## Supported Models

### Flow Prediction (`src/flow/`)

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNET, STGCN, STGODE, UQGNN |
| **Sequence Models** | LSTM, Transformer, PatchTST |
| **Mamba State-Space** | Mamba, Mamba2, Mamba3, Mamba4, Mamba5, Mamba6, Mamba7 |
| **LLM-Based** | STLLM, STLLM2 |
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

# OD data (format: NNDT = Origin x Destination x Time x Features)
python utils/generate.py --data_path /path/to/raw.npy --dataset nyc_taxi_od --years 2018 --fmt NNDT
```

**Supported dimension formats**: `NDT` (Time×N×D), `NTD` (N×T×D), `NT` (N×T, auto-appends D=1), `NNDT` (OD matrix)

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

# With Conformal Quantile Regression
python src/flow/gwnet/main.py --dataset NYISO --years 2018 --quantile --quantile_alpha 0.1

# Test mode
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test

# Export predictions
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test --export

# Organize results into a project folder
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --proj MyExperiment
```

### HPC (Slurm)

```bash
# Submit a Slurm job
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
3. **Load data**: `TimeSeriesDataset` → `DataLoader` → `LoaderAdapter`
4. **Setup callback**: Model-specific preprocessing (adjacency matrices, etc.)
5. **Print args**: Grouped display (Data / Model / Training / Quantile / System / Model-specific)
6. **Build model**: Construct model with final arg values
7. **Create engine**: `BaseEngine` or `CQR_Engine` with optimizer and scheduler
8. **Train**: Epoch loop with early stopping, validation, periodic test evaluation
9. **Test**: Per-horizon metric evaluation with optional export
10. **Profile**: Hardware info, memory, inference time, FLOPs

### Model Registration Pattern

Each model provides three functions to `run_experiment()`:

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

Optional callbacks: `make_optimizer`, `make_scheduler`, `load_data`, `engine_cls`.

Statistical models (ARIMA, VAR, etc.) use `make_optimizer=NO_OPTIMIZER` to skip gradient-based training, with custom engine classes for their specific training loops.

### OD Model Dimension Handling

OD models treat the N×N origin-destination matrix differently. In `setup()`, they override dimensions:

```python
def setup(args, data_path, adj_path, node_num, device, logger):
    args.input_dim = node_num    # N columns as features
    args.output_dim = node_num   # Output: N columns
```

### Graph Normalization

`utils/graph_algo.py` supports multiple adjacency matrix normalization methods:

| Method | Description | Used by |
| --- | --- | --- |
| `scalap` | Scaled Laplacian | STGCN, ASTGCN |
| `normlap` | Normalized Laplacian | General |
| `symadj` | Symmetric adjacency | D2STGNN |
| `transition` | Random walk | DCRNN |
| `doubletransition` | Bidirectional random walk | GWNET, DGCRN |
| `identity` | Identity matrix | Baselines |

## Result Analysis

```bash
# Compare all models on a result directory (auto-discovers models/datasets)
python utils/res.py --path result/Test

# Filter to specific datasets or models
python utils/res.py --path result/Test --datasets nyc_mobility --models STGCN GWNET

# Detailed summary of a single log file
python utils/res.py --log result/Test/STGCN/nyc_mobility/2026-04-16_09-08-38.log

# Select best log by RMSE instead of MAE
python utils/res.py --path result/Test --select RMSE
```

## Acknowledgements

This repository re-implements open-source spatiotemporal forecasting baselines. Please cite the original authors when using specific models.
