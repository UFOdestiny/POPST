# POPST

POPST (Probabilistic and OD Prediction for SpatioTemporal data) is a unified benchmarking framework for spatiotemporal forecasting. It covers both **node-level flow prediction** (N -> N) and **origin-destination (OD) matrix forecasting** (N x N -> N x N), with a shared training engine, logging system, evaluation pipeline, and Conformal Quantile Regression (CQR) wrapper so that every model can be compared under identical conditions.

## Key Features

- **Unified Engine**: A single `BaseEngine` handles training loops, early stopping, checkpointing, and evaluation for all deep-learning models.
- **Conformal Quantile Regression**: `CQR_Engine` wraps any point-prediction model to produce calibrated prediction intervals with coverage guarantees.
- **Comprehensive Metrics**: MAE, RMSE, MAPE, KL divergence, CRPS, MPIW, WINK, Coverage, Interval Score, and quantile loss.
- **Multiple Scalers**: LogScaler, LogMinMaxScaler, StandardScaler (with OD-matrix mode), MinMaxScaler, and RatioScaler.
- **Cross-Platform**: Runs on both Linux (HPC cluster) and Windows with automatic path and resource detection.

## Repository Layout

```
POPST/
  base/                   Core abstractions
    model.py              BaseModel, QuantileOutputLayer, QuantileRegressor
    engine.py             BaseEngine (training, evaluation, checkpointing)
    CQR_engine.py         Conformal Quantile Regression engine
    metrics.py            All metric functions and the Metrics tracker
  utils/                  Shared utilities
    args.py               Argument parser, path config, set_seed, check_quantile
    dataloader.py         DataLoader, dataset registry, dynamic graph construction
    generate.py           Data generation, scalers (Standard, Log, LogMinMax, MinMax, Ratio)
    graph_algo.py         Graph normalization (symmetric, asymmetric, Chebyshev, scaled Laplacian)
    get_adj_mat.py        Adjacency matrix construction from geographic data
    log.py                Logger with rotating file handler
  src/
    flow/                 Flow prediction models (N -> N)
    od/                   OD matrix prediction models (N x N -> N x N)
  res.py                  Result visualization and analysis
```

## Supported Models

### Flow Prediction (N -> N)

Located in `src/flow/`:

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNET, STGCN, STGODE, UQGNN |
| **Sequence Models** | LSTM, Transformer, PatchTST, Mamba (variants 1-7) |
| **LLM-Based** | STLLM, STLLM2 |
| **Probabilistic** | GluonTS |
| **Baselines** | HL (Historical Last), STTN |

### OD Matrix Prediction (N x N -> N x N)

Located in `src/od/`:

| Category | Models |
| --- | --- |
| **Graph Neural Networks** | AGCRN, ASTGCN, GWNET, MPGCN, STGCN, STGODE |
| **Sequence / MLP** | LSTM, STTN, ODMixer |
| **Specialized OD** | STZINB, GMEL, HMDLF, MYOD |
| **Statistical** | ARIMA, SARIMA, VAR |
| **Baselines** | HA (Historical Average), HL (Historical Last) |

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch >= 1.10
- NumPy, SciPy, statsmodels
- Optional: `geopandas`, `shapely` (for adjacency matrix generation), `psutil` (for Windows memory tracking)

### Data Preparation

Datasets should follow this directory structure:

```
<data_root>/
  <dataset_name>/
    adj.npy                 # Adjacency matrix (N x N)
    <year>/
      his.npz               # Normalized data tensor + optional scaler params (min, max)
      idx_train.npy          # Training sample indices
      idx_val.npy            # Validation sample indices
      idx_test.npy           # Test sample indices
```

Use `utils/generate.py` to create `his.npz` and index files from raw data. Use `utils/get_adj_mat.py` to build adjacency matrices from geographic shapefiles.

Configure the dataset root path in `utils/args.py` by editing `_SYSTEM_CONFIG`.

### Supported Datasets

The framework includes configurations for: NYISO, CAISO, NYC, Chicago, Shenzhen, Tallahassee, SafeGraph (FL/CA/TX/NY), and various NYC/Shenzhen OD datasets (taxi, bike, subway).

### Usage

Each model is self-contained in its own directory. To train a model, run its `main.py`:

```bash
# Flow prediction
python src/flow/stgcn/main.py --dataset NYISO --years 2018

# OD prediction
python src/od/gwnet/main.py --dataset nyc_taxi_od --years 2018

# With Conformal Quantile Regression
python src/flow/gwnet/main.py --dataset NYISO --years 2018 --quantile --quantile_alpha 0.1

# Test mode with a saved model
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test --model_path /path/to/model.pt

# Export predictions
python src/flow/stgcn/main.py --dataset NYISO --years 2018 --mode test --export
```

### Common Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `CAISO` | Dataset name (must match `get_dataset_info` registry) |
| `--years` | `2018` | Data sub-folder for the time period |
| `--seq_len` | `24` | Input sequence length |
| `--horizon` | `6` | Prediction horizon |
| `--bs` | `64` | Batch size |
| `--max_epochs` | `2000` | Maximum training epochs |
| `--patience` | `30` | Early stopping patience |
| `--quantile` | `True` | Enable CQR prediction intervals |
| `--quantile_alpha` | `0.1` | Significance level for prediction intervals |
| `--mode` | `train` | `train` or `test` |
| `--export` | `False` | Save predictions as `.npy` file |
| `--seed` | `2026` | Random seed |

## Results

Experiment logs, model checkpoints (`.pt`), and exported predictions (`.npy`) are saved to the path configured in `_SYSTEM_CONFIG`. Use `res.py` to load and visualize exported results.

## Architecture

```
main.py  -->  BaseEngine / CQR_Engine  -->  BaseModel (subclass)
   |               |                            |
   |          train / evaluate              forward(x)
   |               |                            |
   +--- DataLoader + Scaler             QuantileOutputLayer (optional)
   |               |
   +--- Metrics    +--- save_model / load_model
```

- **BaseModel** provides `param_num()`, `horizon` property, and dimension tracking (`seq_len`, `node_num`, `input_dim`, `output_dim`).
- **BaseEngine** handles the training loop with gradient clipping, learning rate scheduling, early stopping, GPU/CPU memory logging, and per-horizon test evaluation.
- **CQR_Engine** extends any engine with conformal calibration: it fits nonconformity scores on a calibration set and adjusts quantile bounds for valid coverage.

## Acknowledgements

This repository re-implements open-source spatiotemporal forecasting baselines. Please cite the original authors when using specific models.
