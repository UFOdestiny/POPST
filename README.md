# POPST

PyTorch forecasting for origin–destination demand and node flow. The retained PDR
method is **PDRHurdleQuantile** (`od/pdr_hurdle`). It predicts flow occurrence and
positive-demand quantiles, then returns the marginal median in count space.

## Run

Use Python 3.11 with the project dependencies:

```bash
pip install --no-build-isolation -r requirements.txt
python run.py config/runs/dc_od_60min_bike/od_pdr_hurdle.yaml --dry-run
python run.py config/runs/dc_od_60min_bike/od_pdr_hurdle.yaml --device cuda:0
python run.py config/suites/od_pdr_hurdle.yaml
```

The method suite contains six datasets: DC Bike, Chicago Bike/Taxi/TNP, and NYC
Manhattan FHV/Taxi. It schedules one experiment per visible GPU and reuses completed
matching runs. Baselines are available through `config/suites/od_baselines.yaml`;
existing checkpoints can be evaluated without training them again.

```bash
python run.py --run results/<experiment>/<run-id> --set runtime.mode=test
```

## Original protocol

| Setting | Value |
|---|---|
| Data | Existing `2025_12to1` arrays and `legacy` splits/scaler |
| History / horizon | 12 / 1 |
| Seed / batch size | 2026 / 128 |
| Epoch limit / patience / min_delta | 400 / 30 / 0.001 |
| Evaluation | Complete test split, original count scale |

The configuration loader accepts the original data protocol. Formal comparisons
use native model outputs without additional thresholding or rounding. MAE, MSE,
RMSE and MAPE measure count errors; F1 detects `prediction > 0`, and TZR measures
`P(prediction <= 0 | target == 0)`. MAPE excludes zero targets.

PDRHurdleQuantile's encoder, dataset, training prior, decoder and loss are contained
in [models/od/pdr_hurdle.py](models/od/pdr_hurdle.py). Its occurrence probability
selects zero at `q <= 0.5`; otherwise it queries the positive quantile at
`1 - 0.5/q`. Quantiles are monotonic, with interpolation and constant tails.

## Results and paths

Each run saves its resolved configuration, status, logs, `metrics.json` and neural
`best.pt` under `results/<experiment>/<run-id>/`. Suite logs are in `results/suites/`.
The complete six-metric method/ODMixer comparison is in
[original-protocol-full-comparison.csv](benchmarks/original-protocol-full-comparison.csv).
The [all-model table](benchmarks/original-protocol-all-models.csv)
contains retained baseline results; uncomputed F1/TZR values are empty.

Refresh the all-model table with
`python tools/report.py`.

The method improves MAE, MAPE, F1 and TZR over original ODMixer on all six datasets.
RMSE/MSE improve on Chicago Bike and Taxi. These results do not establish overall SOTA.

Copy `.env.example` to `.env` to configure dataset, output, cache and pretrained-model
paths. Shared datasets are read only; generated outputs remain local. See the
[configuration guide](config/README.md), [run catalog](config/runs/README.md) and
[OD models](models/od/README.md).

`bash run.sh` runs the method suite. `sbatch run.sh` requests the resources specified
in that script; `devices: all` uses the GPUs visible inside the allocation.

## Checks

```bash
python -m pytest -q
ruff check run.py config data engine models tests --select F401,F841,F811,F821
bash -n run.sh
```
