# Run catalog

Each `config/runs/<dataset_id>/<run>.yaml` selects the dataset named by its parent folder.
Model defaults stay in `config/models/`. See the [configuration guide](../README.md).

| Dataset group | Runs |
|---|---|
| `dc_60min`, `nyc_manhattan_15min`, `chicago_15min`, `sf_15min` | Flow models |
| `dc_od_60min`, `nyc_manhattan_od_15min`, `chicago_od_15min`, `sf_od_15min` | City OD baselines and PDR variants |
| `dc_od_60min_bike` | GPU baselines, PDR variants, CPU references, and calibration |
| `nyc_manhattan_od_15min_fhv`, `nyc_manhattan_od_15min_taxi` | GPU baselines and PDR variants |
| `chicago_od_15min_tnp`, `chicago_od_15min_taxi`, `chicago_od_15min_bike` | GPU baselines and PDR variants |

```bash
python run.py config/runs/chicago_od_15min_taxi/od_agcrn.yaml --device cuda:2
```

Direct-run output names include the dataset ID. Suites supply their own experiment names.
Each execution gets a unique directory; existing results and checkpoints remain in place.

## OD suites

Both suites use the six single-channel datasets above, seed 2026, and all visible GPUs:

| Suite | Models per dataset | Tasks |
|---|---|---|
| `od_baselines.yaml` | 10 neural baselines | 60 |
| `od_zeropdr.yaml` | PDR, PDR_REG, four structural ablations, and three regression distribution heads | 54 |

The baseline suite starts with the six HL runs, followed by AGCRN, ASTGCN, GWNET,
STGCN, STGODE, STZINB, ODMixer, STPro, and OD-CED_OD. The four city OD suites and four
city Flow suites each contain 19 tasks. Flow suites use `flow_hl.yaml` as their
dataset run and select models through `matrix.model.id`.

The PDR variants are `pdr`, `pdr_reg`, `pdr_no_context`, `pdr_no_zone_embed`,
`pdr_no_spatial`, `pdr_no_moe`, `pdr_reg_gau`, `pdr_reg_lap`, and `pdr_reg_t`.
Structural ablations use ZINB/NLL and should be compared with `pdr`; they are not
regression-head ZeroCal ablations. Neither suite includes CPU models or checkpoint-dependent
calibration. CPU reference recipes remain available as individual runs. `run.sh`
executes the baseline suite followed by the PDR suite and forwards arguments to both.

## Evaluation and calibration

Use an exact source run; the launcher never guesses the latest checkpoint:

```bash
python run.py --run results/dc_od_60min_bike_od_pdr_reg/<run-id> --set runtime.mode=test
python run.py config/runs/dc_od_60min_bike/od_split.yaml \
  --run results/dc_od_60min_bike_od_agcrn/<run-id> --set model.id=od/agcrn
python run.py config/runs/dc_od_60min_bike/od_aci.yaml \
  --run results/dc_od_60min_bike_od_agcrn/<run-id> --set model.id=od/agcrn
python run.py config/runs/dc_od_60min_bike/od_zero_cal.yaml \
  --run results/dc_od_60min_bike_od_pdr_reg/<run-id> \
  --set calibration.options.zero_cqr_period=24
```

With a YAML and `--run`, the YAML selects configuration and `--run` supplies `best.pt`.
Dataset, version, channels, and architecture parameters must match the checkpoint.
`od/pdr_reg_post` loads PDR_REG weights for ZeroCal. Use period 24 for hourly data;
the existing recipe defaults to 96. For other datasets, add a calibration recipe in its
dataset folder with the correct `data.id` and checkpoint. Never share one checkpoint
across different models or datasets.

OD Split CP calibrates absolute point residuals. Gaussian/Laplace/Student-t residual
intervals are not native parametric intervals or distributional CRPS. The complete ZeroCal
causal and metric audit remains unfinished. Publication comparisons also require regenerated
train-scaled data and validation-only tuning; see the [baseline audit](../../models/od/README.md).
