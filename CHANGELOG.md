# Changelog

## 2.0 — 2026-09-29

- Replace Flow LSTM/Transformer and OD LSTM/HMDLF/STTN registrations and recipes
  with the current model catalog. Add STPro_OD and history-only OD-CED_OD;
  update ODMixer branches and supervised losses. Retain upstream STPro licensing.
- Use HL entry recipes for city Flow suites. OD baselines schedule 60 tasks;
  PDR schedules 54. `run.sh` executes both suites sequentially, stopping on failure.
- Standardize training defaults to batch size 128, at most 400 epochs, patience 30,
  and validation `min_delta` 0.001; preserve smaller improvements in best checkpoints.
- Reuse successful experiments by comparing saved configuration values directly.
  Support forcing new runs with `runtime.skip_completed: false`.
- Remove OD-CED parameters that never participate in prediction or training, and
  DCRNN's unused concatenation helper and unimplemented fully connected gate path.
  OD-CED checkpoints with the removed keys require retraining for strict loading;
  seeded initialization and parameter counts change.
- Update configuration, run, baseline provenance and execution documentation.

## 1.0

The original v1.0 commit is preserved as the parent of v2.0.
