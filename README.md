# POPST

POPST (Probabilistic and OD Prediction for SpatioTemporal data) is a unified benchmarking framework for spatiotemporal forecasting. It covers both **node-level flow prediction** and **origin-destination (OD) matrix forecasting**, with shared runners, shared evaluation/metric infrastructure, optional **flow matching**, and Conformal Quantile Regression (CQR).

## Key Features

- **Shared Runner**: `run_experiment()` in `base/runner.py` — every model's `main.py` reduces to a config dict + model class + one function call.
- **Unified Engines**: `BaseEngine`, `FlowMatchingEngine`, and `CQR_Engine` share the same runner, checkpointing, logging, and metric pipeline.
- **Shared Flow Matching Integration**: `engine_mode=flow_matching` reuses standard flow models through `base/fm_model.py` and runs them with `base/fm_engine.py`.
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
    fm_engine.py          FlowMatchingEngine for shared flow-matching runs
    fm_model.py           Shared wrapper that adds FM hooks to standard flow models
    CQR_engine.py         Conformal Quantile Regression engine
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
    flow/                 Flow prediction models (26 models)
    od/                   OD matrix prediction models (17 models)
  bash/                   Experiment scripts organized by region (CA, FL1, FL2, NY)
  jobs/                   Slurm job scripts for HPC
  requirements.txt        Python dependencies with pinned versions
```

## Supported Models

### Flow Prediction (`src/flow/`)

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, FMGCN, GWNET, STGCN, STGODE, UQGNN |
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

# With flow matching engine (wraps a standard flow model with the shared FM wrapper)
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --engine_mode flow_matching

# Dedicated FM graph backbone (AGCRN-style recurrent graph model + shared FM engine)
python src/flow/fmgcn/main.py --dataset chicago_mobility --years 2025 --engine_mode flow_matching --proj Chi_Mobi_15min_FM

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

# Submit the dedicated Chi FMGCN run
sbatch jobs/fmgcn.sh
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

### Engine Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--engine_mode` | `standard` | Engine path to use: `standard` or `flow_matching` |
| `--fm_flow_weight` | `0.1` | Weight of the residual vector-field loss used by `FlowMatchingEngine` |
| `--fm_ode_steps` | `20` | Euler integration steps used for sampled FM decoding |
| `--fm_num_samples` | `10` | Number of sampled predictions averaged at validation/test/export time |
| `--fm_context_dim` | `32` | Hidden size of the history encoder in `base/fm_model.py` |
| `--fm_time_dim` | `32` | Time embedding size used by the shared FM wrapper |
| `--fm_hidden_dim` | `64` | Hidden size of the shared FM vector-field head |
| `--fm_node_emb_dim` | `8` | Node embedding size used by the shared FM wrapper (`0` disables it) |
| `--fm_output_activation` | `relu` | Post-sampling activation applied before FM metrics/export |

`--quantile` currently only supports `--engine_mode standard`.

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

For OD models that reshape input dimensions, `setup()` can override these values after auto-fill.

## Architecture

### Execution Flow

1. **Parse arguments**: CLI args + model-specific args merged
2. **Auto-fill**: `seq_len`, `horizon`, `input_dim`, `output_dim` from `info.json`
3. **Load data**: `TimeSeriesDataset` → `DataLoader` → `LoaderAdapter`
4. **Setup callback**: Model-specific preprocessing (adjacency matrices, etc.)
5. **Print args**: Grouped display (Data / Model / Training / Engine / Quantile / System / Model-specific)
6. **Build model**: Construct the requested backbone with final arg values
7. **Wrap for FM if needed**: `engine_mode=flow_matching` inserts `FlowMatchingWrapper` for standard flow models
8. **Create engine**: `BaseEngine`, `FlowMatchingEngine`, or `CQR_Engine`
9. **Train**: Epoch loop with early stopping and validation-based checkpoint selection
10. **Test / Export**: Final per-horizon evaluation with optional result export
11. **Profile**: Hardware info, memory, inference time, FLOPs

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

For shared flow matching support, standard flow models do **not** implement a separate FM model class; they are wrapped automatically by the runner when `--engine_mode flow_matching` is used.

`src/flow/fmgcn` is the exception: it is a dedicated graph backbone tuned for the FM path, while still using the shared `FlowMatchingEngine` and wrapper pipeline.

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
python utils/res.py --path result/MyExperiment

# Filter to specific datasets or models
python utils/res.py --path result/MyExperiment --datasets nyc_mobility --models STGCN GWNET

# Detailed summary of a single log file
python utils/res.py --log result/MyExperiment/STGCN/nyc_mobility/2026-04-16_09-08-38.log

# Select best log by RMSE instead of MAE
python utils/res.py --path result/MyExperiment --select RMSE
```

## Acknowledgements

This repository re-implements open-source spatiotemporal forecasting baselines. Please cite the original authors when using specific models.
