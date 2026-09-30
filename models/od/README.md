# OD models

Models consume OD histories `(B,T,O,D,C)` and evaluate predictions in original
count space. Each model owns its architecture and method-specific setup; the
shared engine provides configuration, checkpointing and metrics.

## PDRHurdleQuantile

`od/pdr_hurdle` is implemented in [pdr_hurdle.py](pdr_hurdle.py). It contains its
own shared current/previous-day encoder, causal dataset, loader, training-only
activity prior, occurrence head, monotonic positive quantiles and training loss.
It does not import another forecasting model.

Training combines current and previous-day count MAE with occurrence BCE and
positive-target pinball loss, both weighted 0.1. Origin/destination features keep
occupancy and mean count per active edge separate. The point forecast is the
approximate marginal median: zero for `q <= 0.5`, otherwise the positive quantile
at `1 - 0.5/q`. Occurrence is detached from the point decision during training.

Default encoder: hidden width 16, five layers, dropout 0.3. There are 11 positive
quantiles from 0.01 to 0.99 with linear interpolation and constant tails. The
method uses the original scaler/splits, Adam at 0.001 with epsilon 1e-12,
StepLR(200, 0.95), batch 128 and seed 2026.

## Baselines

| Model IDs (`od/` prefix) | Implementation |
|---|---|
| `zero`, `persistence`, `seasonal`, `ha` | Zero, last observation, previous daily cycle, rolling mean |
| `arima`, `sarima`, `var`, `hl` | Causal statistical predictors |
| `agcrn`, `gwnet`, `astgcn`, `stgcn`, `stgode` | Graph models adapted to OD channels |
| `stzinb` | ZINB model adapted to OD |
| `odmixer` | Current/previous-day branches, shared encoder, bidirectional trend learning and both MAE losses |
| `stpro` | Learned O/D prototype projections and dual cross-attention |
| `od_ced` | OD-history STAR embedding and coarse-to-fine prediction |

OD adaptations process mobility channels independently. ODMixer uses complete-count
histories rather than unfinished metro orders. OD-CED uses training-only semantic
coarsening without geographic, POI or calendar inputs. These task adaptations are
not reproductions of the papers' complete benchmark settings.

Statistical parameters fit the training prefix; subsequent observed history updates
state without refitting. STGODE semantic graphs and OD-CED coarsening use training
observations. Model-specific losses differ; count-space metrics are shared.

Source implementations: [ODMixer](https://github.com/KLatitude/ODMixer),
[STPro](https://github.com/AIMS-SDU/STPro), [OD-CED](https://github.com/luckyyangrun/OD-CED),
[AGCRN](https://github.com/LeiBAI/AGCRN), [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet),
[ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch),
[STGODE](https://github.com/square-coder/STGODE), [STZINB](https://github.com/ZhuangDingyi/STZINB).
STGCN uses the [PyTorch port](https://github.com/hazdzz/stgcn).

Checkpoints load strictly. A checkpoint from a different architecture is not a
compatible substitute. Tests cover tensor axes, causal inputs, checkpoint replay,
quantile ordering and training-only priors; benchmark scores are stored separately.
