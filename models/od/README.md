# OD baseline implementation audit

Updated 2026-09-29. Algorithms and model-specific data handling live in each model module;
the engine provides training, evaluation, and reporting.

| Registered model | Implementation / changes | Comparison label |
|---|---|---|
| `zero` | Constant zero at the same test origins as other models | Zero |
| `persistence` | Last observed OD matrix, repeated over the forecast horizon | Persistence |
| `seasonal` | Last observed daily cycle, period inferred from data frequency | Seasonal naive |
| `ha` | Mean of the configured trailing observed window, rolling at each origin | Rolling HA |
| `arima` | Per-cell fit on training observations; fixed parameters, state extended only with observations available at each origin | ARIMA |
| `sarima` | Same rolling protocol, with a nonzero seasonal AR term and daily period by default | SARIMA |
| `var` | Training-only truncated SVD followed by fixed-lag VAR, rolling with observed latent history | Low-rank VAR |
| `hl` | Existing local historical linear predictor | HL |
| `agcrn` | Corrected horizon/destination output axes | AGCRN, adapted to OD |
| `gwnet` | Corrected horizon/destination axes; graph supports move with the model | Graph WaveNet, adapted to OD |
| `astgcn` | Corrected temporal stride sizing and device-safe Chebyshev supports | ASTGCN recent branch, adapted to OD |
| `stgcn` | Corrected single-step temporal head; invalid empty temporal head rejected | STGCN, adapted to OD |
| `stgode` | Semantic graph uses training-only daily raw-count profiles and DTW; corrected temporal residual block | STGODE, adapted to OD |
| `stzinb` | Preserved temporal/node axes in temporal convolutions; explicit multi-horizon ZINB outputs | STZINB, adapted to OD |
| `odmixer` | Current and previous-day input branches, shared encoder, BTL and both supervised branch losses; dataset and engine are in `odmixer.py` | ODMixer_OD |
| `stpro` | Author release's learned O/D prototype projections and dual cross-attention; repaired dimensions and attention layout | STPro_OD |
| `od_ced` | Author release's STAR embedding, coarse-to-fine decoder and convolutional heads; training-only semantic coarsening, learned horizon queries | OD-CED_OD (OD history only) |

OD adaptations process mobility channels independently; their presence in this repository
does not establish equivalence to their original node-demand experiments.

ARIMA/SARIMA report fit failures instead of silently producing zero-scored NaNs. A training
series that is constant uses its training constant as an explicitly defined fallback.
Their order selection still requires validation-only tuning for publication experiments.

## Source evidence

Source inspections used author papers/repositories where available:

- [AGCRN author implementation](https://github.com/LeiBAI/AGCRN).
- [Graph WaveNet author implementation](https://github.com/nnzhan/Graph-WaveNet).
- [ASTGCN author implementation](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch).
- [STGCN PyTorch port](https://github.com/hazdzz/stgcn), a third-party port rather than the original author code.
- [STGODE author implementation](https://github.com/square-coder/STGODE).
- [STZINB author implementation](https://github.com/ZhuangDingyi/STZINB), originally a node-demand model.
- [ODMixer paper](https://arxiv.org/abs/2404.15734) and [author implementation](https://github.com/KLatitude/ODMixer).
  The paper's metro task uses unfinished-order preprocessing. Our complete-count OD datasets
  do not provide those inputs: this integration preserves the two-branch mechanism but is
  explicitly an OD task adaptation. Multi-horizon heads are a local extension.
- [STPro paper, IJCAI 2025](https://www.ijcai.org/proceedings/2025/400) and
  [author repository](https://github.com/AIMS-SDU/STPro). The inspected public source imports
  an absent `model/AGCRNCell.py`; its `Dual.forward` accesses undeclared `node_num`/`input_dim`
  and includes dataset-specific fixed dimensions. This adapter removes unused broken imports,
  parameterizes history/node dimensions, repairs attention head axes, and pads/crops the
  prediction head for node counts that are not multiples of three. It follows the active
  learned prototype projection in the release, not the commented clustering variants.
- [OD-CED paper](https://arxiv.org/html/2503.24237v1) and
  [author implementation](https://github.com/luckyyangrun/OD-CED). The public repository
  supplies external region assignments but no preprocessing script. The OD-only adapter
  selects dense seeds by training OD volume and propagates their labels through semantic
  OD transitions. It omits geographic transitions, POI and calendar inputs. Every coarse
  matrix is computed by summing raw counts before applying the project scaler. Learned
  forecast-step queries replace the release's clock embeddings. The released STAR sum,
  cross-attention decoder, fine/coarse convolutional heads and masked dual MSE are retained;
  the bypassed encoder, decoder self-attention, induce attention and unused head
  normalization have been removed from the model. This is
  **OD-CED_OD**, not a reproduction of the complete paper's method or reported results.

## Training adaptations

All three integrations use project windows `(B,T,O,D,C)`, independent mobility channels,
the shared scaler and splits, and whole-matrix count-space validation MAE for checkpoint
selection. The global defaults are at most 400 epochs, patience 30, and a significant
validation decrease of at least 0.001. Test observations never fit coarsening or model
parameters. Prediction heads return unrestricted point estimates, as other project
regression models do; the OD-CED release's evaluation-only ReLU is not applied.

ODMixer retains count-space MAE on both branches and uses the released 16 hidden units,
five layers and Adam epsilon. STPro uses the released 128-dimensional latent state and
two attention heads, with Xavier initialization. OD-CED uses the released masked fine
dense-pair and coarse sparse-group MSE in count space, Adam betas `(0.5,0.999)` and
15-epoch learning-rate decay. These training losses differ, but all models report the
same evaluation metrics. ODMixer's release-specific MAPE scheduler is replaced by the
repository's configured StepLR; all hyperparameters are recorded in each resolved config.

Run the history-only OD-CED integration with:

```bash
python run.py config/runs/dc_od_60min_bike/od_od_ced.yaml --device cuda:0
```

OD-CED checkpoints created before this cleanup contain unused parameter keys and
need retraining for strict loading by this version. Removing those parameters also
changes seeded initialization; prior metrics should not be reused as fresh results.

## Evaluation repairs and validation

Neural checkpoints are selected on validation; test runs once afterward. Use
`training.evaluate_test: false` during tuning. Validation losses are weighted by label
count, and invalid predictions cannot become zero errors.

Statistical models use the same forecast origins and horizons as neural models. Parameters
fit the training prefix; later observed history updates state without refitting. ACI waits
until each target is observed before using its residual. Its bounded controller is a local
adaptation without a claim to the original algorithm's theorem.

The current OD baseline catalog contains 17 models (10 neural and seven CPU).
Small-fixture training/reload checks are in `tests/test_execution.py`; model-specific
checks are in `tests/test_od_baselines.py`. Regression tests cover multi-horizon/channel
axes, batch ordering, ODMixer's past branch, causal
history, split boundaries, and training-only scaling. These verify execution, not paper
reproduction accuracy or SOTA.

On 2026-09-29, all 107 repository tests passed. The three author-code adapters also
completed one-epoch CPU smoke runs on the existing DC bike and DC two-channel datasets
(four samples per split). ODMixer's single-horizon predictions matched both upstream
branches exactly after loading identical weights, for batch sizes one and two. These
short runs are execution checks, not benchmark results; source snapshots and the smoke
ledger are cached under `artifacts/baseline_sources/`.

Formal comparisons still require regenerated train-scaled data, validation-only tuning,
review-driven uncertainty metrics, and a complete ZeroCal causal audit. Existing shared
`2025_12to1` arrays are unchanged; full benchmark training has not been performed.
