# POPST

POPST is a working collection of reproducible spatiotemporal forecasting baselines. It covers both node-level flow prediction (N->N) and origin-destination (OD) matrix forecasting (N x N) and shares one training engine, logger, and evaluation pipeline so every model can be compared with the exact same data processing, metrics, and artifacts.

## Repository Layout
| Path | Purpose |
| --- | --- |
| `src/flow/` | Flow (N->N) baselines (AGCRN, ASTGCN, DCRNN, DGCRN, D2STGNN, DSTAGNN, GWNET, HL/HA, LSTM, Mamba, STGCN, STGODE, STTN, Transformer, UMamba, UQGNN, ...). |
| `src/od/` | OD (N x N) baselines (AGCRN, ARIMA/SARIMA/VAR, ASTGCN, GMEL, GWNET, HA/HL, HMDLF, LSTM, MPGCN, MYOD, ODMixer, STGCN, STGODE, STTN, STZINB, ...). |
| `src/archived/` | Frozen experiment variants (e.g., PGNN, STGCN Gaussian/Laplace, TrustEnergy) kept for reference. |
| `base/` | Core abstractions (`BaseModel`, `BaseEngine`, `Quantile_Engine`, metrics, shared models). |
| `utils/` | Argument parser, dataloaders, scalers, dataset generators, adjacency builders, logging utilities. |
| `datasets/` | Processed datasets laid out as `<dataset>/<year>/{his.npz,idx_*.npy}` plus top-level `adj.npy`. |


## Supported Baselines
- **Flow (N->N)**: AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNET, HA/HL, LSTM, Mamba/Mamba2, STGCN, STGODE, STTN, Transformer, UMamba/UMamba2, UQGNN.
- **Origin-Destination (N x N)**: AGCRN, ARIMA, ASTGCN, GMEL, GWNET, HA/HL, HMDLF, LSTM, MPGCN, MYOD, ODMixer, SARIMA, STGCN, STGODE, STTN, STZINB, VAR.
- **Statistical & probabilistic tools**: Negative binomial heads (STZINB), ARIMA/SARIMA/VAR, conformal-quantile wrappers, CRPS/KL metrics, coverage diagnostics.

## Dataset Specification
Every dataset entry follows the same contract:

```
datasets/
	<dataset_name>/
		adj.npy                # adjacency used by graph-based models
		<year>/
			his.npz              # normalized tensor (time, nodes[, nodes2], channels)
			idx_train.npy        # sample indices for seq2seq windows
			idx_val.npy
			idx_test.npy
			idx_all.npy
```

- `his.npz` contains the log-scaled sequences and (optionally) `min`/`max` for `LogMinMaxScaler`.
- The dataloader reads `years` from `--years` and expects `seq_len` historic steps and `horizon` targets.
- `utils.args.get_data_path()` points to the dataset root; edit `_SYSTEM_CONFIG` (or symlink your data directory) before running if your layout differs from the default Linux/Windows paths.

### Provided dataset keys
**Flow (node-level)**

| Key | #Regions | Notes |
| --- | --- | --- |
| `panhandle` | 924 | SafeGraph mobility for Florida panhandle counties. |
| `Shenzhen`, `Shenzhen2` | 491 | Hourly Shenzhen flows (different preprocessing versions). |
| `NYC`, `NYC_Crash`, `NYC_Combine` | 67 / 42 | NYC crowd flow variants (raw, crash-only, combined signals). |
| `Chicago` | 77 | Chicago regional flows. |
| `NYISO`, `CAISO`, `Tallahassee` | 11 / 9 / 9 | Power load benchmarks. |
| `NYISO_HDM`, `CAISO_HDM`, `Tallahassee_HDM` | 11 / 9 / 9 | Hour-day-month augmented versions. |

**Origin-Destination matrices**

| Key | Size | Notes |
| --- | --- | --- |
| `nyc_taxi_od`, `nyc_bike_od`, `nyc_subway_od` | 67x67 | Manhattan taxi, bike, subway OD tensors. |
| `nyc_subway_bike_od`, `nyc_subway_taxi_od` | 67x67 | Cross-modality OD transfer datasets. |
| `sz_taxi_od`, `sz_bike_od`, `sz_subway_od`, `sz_dd_od` | 491x491 | Shenzhen taxi/bike/subway/Didi OD benchmarks. |
| `sz_subway_bike_od`, `sz_subway_taxi_od`, `sz_taxi_bike_od` | 491x491 | Cross-modality Shenzhen OD variants. |

Add additional datasets by editing `utils/dataloader.get_dataset_info()` so the loader knows where to find `his.npz` and `adj.npy`.


## Acknowledgements
This repo re-implements open-source baselines released by the original authors. Please cite their papers when you use specific models, and cite POPST (this repository) if the unified training pipeline or prepared datasets help your research.