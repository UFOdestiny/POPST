"""Replay an exact checkpoint with the original protocol and native decoder."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from config.loader import load_config, resolved_copy
from config.paths import load_paths
from engine.logging import get_logger
from engine.runner import build_engine

OUT = ROOT/'results/replays'


def evaluate(run, device):
    run = Path(run).resolve()
    if json.loads((run/'status.json').read_text())['status'] != 'succeeded':
        raise ValueError('A completed checkpoint is required')
    cfg = load_config(run/'config.resolved.yaml')
    if (cfg.data.version != '2025_12to1' or cfg.training.batch_size != 128
            or cfg.training.max_epochs != 400 or cfg.training.patience != 30
            or cfg.training.min_delta != .001 or cfg.training.seed != 2026
            or any(getattr(cfg.runtime, f'max_{s}_samples') is not None for s in ('train', 'val', 'test'))):
        raise ValueError('Checkpoint does not use the original unified protocol')
    stem = f'{cfg.data.id}-{cfg.model.id.split("/")[-1]}'
    directory = OUT/stem
    cfg = resolved_copy(cfg, {'runtime': {'mode': 'test', 'device': device}})
    logger = get_logger(str(directory), stem, log_filename='evaluation.log')
    cfg, engine, _, _, _ = build_engine(cfg, load_paths(), directory, logger)
    counts = np.zeros(4, dtype=np.int64)
    original = engine.metric.compute_one_batch

    def capture(preds, labels, null_val, mode='train', **kw):
        if mode != 'test':
            raise RuntimeError('This script only evaluates frozen checkpoints')
        valid = torch.isfinite(labels)
        if not torch.isfinite(preds[valid]).all() or (labels[valid] < 0).any():
            raise ValueError('Invalid count-space forecast or target')
        predicted, actual = preds > 0, labels > 0
        counts[:] += [(valid & predicted & actual).sum().item(),
                      (valid & predicted & ~actual).sum().item(),
                      (valid & ~predicted & actual).sum().item(),
                      (valid & ~predicted & ~actual).sum().item()]
        return original(preds, labels, null_val, mode, **kw)

    engine.metric.compute_one_batch = capture
    engine.evaluate('test', str(run/'best.pt'), export=False)
    scores = engine.metric.last_test['average']
    saved = json.loads((run/'metrics.json').read_text())['average']
    for name in ('MAE', 'RMSE', 'MSE', 'MAPE'):
        if not np.isclose(scores[name], saved[name], rtol=1e-5, atol=1e-6):
            raise RuntimeError(f'{name} differs from the saved original-protocol result')
    tp, fp, fn, tn = (int(x) for x in counts)
    result = {'dataset': cfg.data.id, 'model': cfg.model.id, 'source': str(run.relative_to(ROOT)),
              **scores, 'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
              'TZR': tn/(tn+fp) if tn+fp else 0, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
    (OUT/f'{stem}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.run, args.device)
