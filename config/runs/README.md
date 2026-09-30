# Run catalog

Each `config/runs/<dataset>/<run>.yaml` selects the dataset named by its parent
folder. Model defaults are in `config/models/`.

| Datasets | Available runs |
|---|---|
| DC, Chicago, NYC Manhattan and SF flow | Node-flow models |
| City OD datasets | OD baselines |
| DC Bike, Chicago Bike/Taxi/TNP, NYC Manhattan FHV/Taxi | OD baselines and PDRHurdleQuantile |

```bash
python run.py config/runs/chicago_od_15min_taxi/od_pdr_hurdle.yaml --device cuda:2
python run.py config/suites/od_pdr_hurdle.yaml
```

The method suite contains six runs of `od/pdr_hurdle`. The baseline suite contains
60 runs: six datasets with HL, AGCRN, ASTGCN, Graph WaveNet, STGCN, STGODE, STZINB,
ODMixer, STPro and OD-CED. Both use the original data protocol and seed 2026.
`run.sh` executes the method suite; baseline execution is a separate command.

## Evaluation

```bash
python run.py --run results/<experiment>/<run-id> --set runtime.mode=test
```

`--run` supplies an exact `best.pt`; it never guesses the newest checkpoint. When
combined with a YAML, that YAML supplies the configuration. Dataset, data version,
channels and architecture must match the checkpoint. Outputs and complete resolved
settings are stored under `results/`.

For frozen-checkpoint F1/TZR and regression-metric verification:

```bash
python tools/evaluate.py \
  --run results/<experiment>/<run-id> --device cuda:0
```

The script uses the native decoder, checks the original protocol and saved metrics,
and writes full-test confusion counts under `results/replays/`.
