"""Collect retained original-protocol results and exact replay metrics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from config.loader import load_config

BUILD = ROOT/'benchmarks'


def report():
    primary = pd.read_csv(BUILD/'original-protocol-full-comparison.csv')
    classification = {r['source']: r for r in primary.to_dict('records')}
    for p in (ROOT/'results/replays').glob('*.json'):
        r = json.loads(p.read_text())
        classification[r['source']] = r
    rows = []
    for path in sorted((ROOT/'results').glob('*/*/metrics.json')):
        run = path.parent
        if json.loads((run/'status.json').read_text())['status'] != 'succeeded':
            continue
        cfg = load_config(run/'config.resolved.yaml')
        if (cfg.training.batch_size != 128 or cfg.training.max_epochs != 400
                or cfg.training.patience != 30 or cfg.training.min_delta != .001
                or cfg.training.seed != 2026 or not cfg.training.evaluate_test
                or any(getattr(cfg.runtime, f'max_{s}_samples') is not None for s in ('train', 'val', 'test'))):
            raise ValueError(f'Nonoriginal experiment budget: {run}')
        source = str(run.relative_to(ROOT))
        scores = json.loads(path.read_text())['average']
        replay = classification.get(source)
        rows.append({'dataset': cfg.data.id, 'model': cfg.model.id,
                     'role': 'method' if cfg.model.id == 'od/pdr_hurdle' else 'baseline',
                     **{k: scores[k] for k in ('MAE', 'RMSE', 'MSE', 'MAPE')},
                     'F1': replay['F1'] if replay else None,
                     'TZR': replay['TZR'] if replay else None,
                     'classification_status': 'full_test_replayed' if replay else 'not_replayed',
                     'source': source})
    pd.DataFrame(rows).to_csv(BUILD/'original-protocol-all-models.csv', index=False)
    print(f'{len(rows)} completed original-protocol results; one retained PDR design.')


if __name__ == '__main__':
    report()
