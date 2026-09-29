# PDR_REG_POST / ZeroCQR

`ZeroCQREngine` is a post-hoc conformal method for sparse OD regression.  It
keeps the PDR_REG point forecast as the interval centre, so interval
calibration does not intentionally trade away MAE/MSE.

The key idea is sparse-aware Mondrian calibration:

1. Predictions classified as zero form their own group; their lower bound is
   clipped at zero.
2. Positive predictions are split by prediction magnitude into quantile bins.
3. Each group receives its own split-conformal residual radius, fitted on the
   validation set at the requested coverage level.

This prevents a few high-demand OD pairs from widening intervals for the many
zero/low-demand pairs.  On `nyc_manhattan_od_15min_fhv` with 95% target
coverage, 8 positive bins obtained MPIW **1.120** and COV **96.961%**, compared
with generic OD-CQR MPIW **1.563**, while retaining MAE **0.266**.

Example:

```bash
python src/od/pdr_reg_post/main.py \
  --dataset nyc_manhattan_od_15min_fhv --years 2025_12to1 \
  --proj NYC_OD --mode test --export \
  --model_path result/NYC_OD/PDR_REG/nyc_manhattan_od_15min_fhv/PDR_REG_2026-07-16_03-36-35.pt \
  --zero_cqr_alpha 0.05 --zero_cqr_active_bins 8
```

`--zero_cqr_active_bins` controls the positive-demand granularity; 8 was best
in the current 1/4/8/16-bin comparison.  `--zero_cqr_zero_floor 0.001` turns
near-zero non-negative point forecasts into exact zeros.
