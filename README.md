# POPST

POPST is a working collection of reproducible spatiotemporal forecasting baselines. It covers both node-level flow prediction (N->N) and origin-destination (OD) matrix forecasting (N x N) and shares one training engine, logger, and evaluation pipeline so every model can be compared with the exact same data processing, metrics, and artifacts.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/flow/` | Flow (N->N) baselines. See supported models below. |
| `src/od/` | OD (N x N) baselines. See supported models below. |
| `base/` | Core abstractions (`BaseModel`, `BaseEngine`, `CQR_Engine`, metrics). |
| `utils/` | Argument parser, dataloaders, scalers, dataset generators, adjacency builders, logging utilities. |

## Supported Baselines

### Flow Prediction (Node-level)
Located in `src/flow/`:
- **Graph-based**: AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNET, STGCN, STGODE, STTN, UQGNN
- **Sequence-based**: LSTM, Mamba (and variants 2-7), Transformer, PatchTST, STLLM, STLLM2
- **Traditional**: HL (Historical Last), HA (Historical Average)
- **Other**: GluonTS integration

### Origin-Destination Matrix Prediction
Located in `src/od/`:
- **Deep Learning**: AGCRN, ASTGCN, GWNET, LSTM, MPGCN, STGCN, STGODE, STTN, STZINB, GMEL, HMDLF, ODMixer, MYOD
- **Statistical**: ARIMA, SARIMA, VAR
- **Baseline**: HA (Historical Average), HL (Historical Last)

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- SciPy

### Data Preparation
This repository expects datasets to be stored separately or linked. The default configuration looks for datasets in a specific system path (see `utils/args.py`).

To run with your own data, ensure your data directory follows this structure:

```
dataset_root/
    <dataset_name>/
        adj.npy                # Adjacency matrix (for graph models)
        <year>/
            his.npz            # History data tensor
            idx_train.npy      # Training indices
            idx_val.npy        # Validation indices
            idx_test.npy       # Test indices
```

You can configure the data path in `utils/args.py` by modifying `_SYSTEM_CONFIG` or by passing the path arguments if supported.

### Usage
Each model is self-contained in its own directory within `src/flow` or `src/od`. To train a model, run its `main.py` script.

**Example: Running STGCN on NYC dataset**

```bash
python src/flow/stgcn/main.py --dataset NYC --years 2018
```

**Common Arguments:**
- `--dataset`: Name of the dataset folder.
- `--years`: Sub-folder year for the data.
- `--seq_len`: Input sequence length (default: 24).
- `--horizon`: Prediction horizon (default: 6).
- `--bs`: Batch size (default: 128).
- `--mode`: `train` or `test`.

## Results
Experiment results, logs, and model checkpoints are saved to the directory specified in `_SYSTEM_CONFIG` (defaulting to a `result` or `output` directory outside the source tree or in a local folder).

## Acknowledgements
This repo re-implements open-source baselines released by the original authors. Please cite their papers when you use specific models.
