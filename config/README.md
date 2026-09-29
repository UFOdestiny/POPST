# Configuration

| Location | Responsibility |
|---|---|
| `default.yaml` | Global training and runtime defaults |
| `datasets/` | Prepared-data paths, frequency, channels, and adjacency |
| `models/` | Model parameters and training defaults; `registry.yaml` maps implementations |
| `runs/<dataset_id>/` | One experiment selecting a model, dataset, mode, and overrides |
| `suites/` | Batches of run paths, parameter combinations, and device assignments |

A **run** is one experiment; a **suite** schedules multiple runs. Run directories match
`data.id`. Configuration merges **default → dataset → model → run → CLI overrides**;
unknown fields and incompatible types are rejected. Dataset/model files supply defaults
and are not standalone experiments unless both IDs are selected.

## Training settings

Edit a run to change one experiment, a model file to change that model's defaults, or
`default.yaml` for global defaults. All supplied methods inherit
`training.batch_size: 128`, `training.max_epochs: 400`, `training.patience: 30`,
and `training.min_delta: 0.001`. Validation loss must decrease by at least
`min_delta` relative to the previous best loss to reset patience. Smaller
improvements count toward patience while still updating the best checkpoint.
For example, in a run YAML:

```yaml
model:
  id: od/pdr_reg
data:
  id: dc_od_60min_bike
training:
  batch_size: 128
  max_epochs: 400
  patience: 30
  min_delta: 0.001
  seed: 2026
  optimizer:
    lr: 0.001
runtime:
  device: cuda:0
```

```bash
python run.py config/runs/dc_od_60min_bike/od_pdr_reg.yaml --dry-run
python run.py config/runs/dc_od_60min_bike/od_pdr_reg.yaml \
  --set training.batch_size=128 --device cuda:1
```

`--config PATH` aliases the positional path. `runtime.mode` selects `train`, `test`, or
`calibrate`. Evaluation/calibration requires an exact `runtime.checkpoint` or
`--run results/<experiment>/<run-id>`. See the [run catalog](runs/README.md).
Single runs accept `cpu`, `cuda`, or `cuda:N`; `all` is a suite setting.

`runtime.skip_completed: true` is the default for both single runs and suites.
The launcher compares saved `config.requested.yaml` values under the configured results
root and reuses runs with `status: succeeded` and nonempty final `metrics.json`.
Recorded checkpoints must still exist; profiling runs also require `efficiency.json`.
Experiment names/comments and device/worker/logging settings are ignored when matching;
model, data, training (including batch size and seed), calibration, and output settings
must match. Failed, interrupted, or incomplete runs are executed again. No hashes are used.
Single-run output reports `skipped: true` and the reused directory. Force a new execution
with `--set runtime.skip_completed=false`.

## Suites

Run paths are relative to the suite YAML. A matrix expands combinations; dotted overrides
apply to every run. Matrix values take precedence over overrides for the same field.

```yaml
runs:
  - ../runs/dc_od_60min_bike/od_agcrn.yaml
  - ../runs/dc_od_60min_bike/od_odmixer.yaml
matrix:
  training.seed: [2026]
overrides:
  training.batch_size: 128
  training.max_epochs: 400
devices: all
skip_completed: true
```

This creates two experiments. `all` discovers every visible CUDA GPU and runs one task
per GPU, starting the next task when a slot finishes. It respects Slurm allocations and
`CUDA_VISIBLE_DEVICES`; no visible GPU is an error. Use `[cuda:0, cuda:2]` to select a
subset. This distributes experiments, not one model across multiple GPUs.

```bash
python run.py config/suites/od_baselines.yaml --dry-run
python run.py config/suites/od_zeropdr.yaml
```

All supplied suites use `devices: all`, seed 2026, and GPU models only. Baselines have
60 tasks; PDR has 54; each city OD suite has 19 and each city Flow suite has 19.
Set suite options in YAML: `--set`, `--device`, and `--run` apply only to single runs.
Suite `tasks.json` records reused tasks as `skipped` with their original run directory;
skips count as successful completion. Set top-level `skip_completed: false` to rerun
the suite. Explicit `runtime.skip_completed` in overrides/matrix takes precedence.

Preprocessing settings and raw/asset/processed paths belong to notebooks. `.env` holds
data, results, cache, and pretrained-model locations. Slurm resources belong to `run.sh`;
legacy saved-run `slurm` metadata is ignored when loading checkpoints.
