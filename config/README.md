# Configuration

| Location | Purpose |
|---|---|
| `default.yaml` | Training and runtime defaults |
| `datasets/` | Dataset paths, frequency, channels and adjacency |
| `models/` | Model parameters; `registry.yaml` maps IDs to implementations |
| `runs/<dataset>/` | One experiment |
| `suites/` | Experiment batches and GPU assignments |

Configuration merges **default → dataset → model → run → CLI overrides**. Unknown
fields and incompatible types are rejected. Data uses `protocol: legacy` and the
existing `2025_12to1` arrays. Experiment defaults are seed 2026, batch 128, maximum
400 epochs, patience 30 and min_delta 0.001; model-specific optimizer settings stay
in the model configuration.

```bash
python run.py config/runs/dc_od_60min_bike/od_pdr_hurdle.yaml --dry-run
python run.py config/runs/dc_od_60min_bike/od_pdr_hurdle.yaml --device cuda:1
python run.py --run results/<experiment>/<run-id> --set runtime.mode=test
```

A run selects `model.id` and `data.id`. `runtime.mode` is `train`, `test` or
`calibrate`; evaluation requires an exact checkpoint. `--config PATH` aliases the
positional path. Single runs accept `cpu`, `cuda` and `cuda:N`.

## Suites

```yaml
runs:
  - ../runs/dc_od_60min_bike/od_pdr_hurdle.yaml
matrix:
  training.seed: [2026]
devices: all
```

`devices: all` schedules one experiment per visible GPU and respects Slurm and
`CUDA_VISIBLE_DEVICES`. An explicit list such as `[cuda:0, cuda:2]` selects GPUs.
Run paths are relative to the suite file. Suite overrides and matrix values belong
in YAML; CLI overrides apply to individual runs.

| Suite | Tasks |
|---|---:|
| `od_pdr_hurdle.yaml` | 6 |
| `od_baselines.yaml` | 60 |
| Each city OD suite | 10 |
| Each city flow suite | 19 |

Completed matching runs are reused by default. Matching includes model, data,
training, calibration and output settings; device and logging settings are ignored.
`status.json`, final metrics and the checkpoint must exist. Use
`--set runtime.skip_completed=false` to request a fresh individual run.

`.env` configures filesystem paths. Slurm resources belong to `run.sh`. See the
[run catalog](runs/README.md) for datasets and evaluation.
