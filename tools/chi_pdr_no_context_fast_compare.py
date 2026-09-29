"""Fast, read-only MPIW comparison for the saved Chicago no-context PDR run."""

import json
import logging
import os
import sys
import argparse

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from base.metrics import zinb_mean
from src.od.pdr_no_context.pdr_no_context_model import PDR_no_context
from utils.dataloader import get_dataset_info, load_adj_from_numpy, load_dataset
from utils.generate import reconstruct_scaler
from utils.graph_algo import normalize_adj_mx


DATASET = "chicago_od_15min_taxi"
YEARS = "2025_12to1"
ALPHA = 0.05
BS = 64
SOURCE = os.path.join(ROOT, "result/Chi_OD/PDR_no_context/chicago_od_15min_taxi")
OUT = os.path.join(ROOT, "result/Chi_OD_CALIBRATION_COMPARE")
CHECKPOINT = os.path.join(OUT, "input_checkpoints/PDR_no_context_chicago_od_15min_taxi.pt")


class Args:
    years = YEARS
    seq_len = 12
    horizon = 1
    bs = BS


def inverse(scaler, x):
    return scaler.inverse_transform(x, device="cuda")


def q(scores):
    x = scores[np.isfinite(scores)].reshape(-1)
    rank = min(int(np.ceil((x.size + 1) * (1 - ALPHA))), x.size) - 1
    return np.partition(x, rank)[rank] if x.size else 0.0


def metrics(point, lower, upper, label):
    err = point - label
    width = upper - lower
    alpha = ALPHA
    interval_score = width + (2 / alpha) * np.maximum(lower - label, 0) + (2 / alpha) * np.maximum(label - upper, 0)
    return {
        "MAE": float(np.abs(err).mean()), "MSE": float((err ** 2).mean()),
        "MPIW": float(width.mean()), "IS": float(interval_score.mean()),
        "COV": float(((label >= lower) & (label <= upper)).mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()
    bins = max(args.bins, 1)
    data_path, adj_path, node_num = get_dataset_info(DATASET, YEARS)
    adj = load_adj_from_numpy(adj_path) - np.eye(node_num)
    # Reuse the source test output read-only; only validation predictions must
    # be evaluated to fit post-hoc calibration parameters.
    raw_test = np.load(os.path.join(SOURCE, "PDR_no_context-chicago_od_15min_taxi-res.npy"), mmap_mode="r")
    test_pred, test_label = np.asarray(raw_test[0]), np.asarray(raw_test[1])
    cache_path = os.path.join(OUT, "chi_pdr_no_context_validation_predictions.npz")
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        val_pred, val_label = cache["pred"], cache["label"]
    else:
        model = PDR_no_context(
            A=normalize_adj_mx(adj, "uqgnn")[0], node_num=node_num, input_dim=node_num,
            output_dim=node_num, seq_len=12, horizon=1, context_dim=64,
            zone_embed_dim=16, num_experts=3,
            head_hidden_dim=128, dropout=0.0,
        ).cuda().eval()
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=False))
        loader, scaler = load_dataset(data_path, Args(), logger=logging.getLogger("fast_compare"))
        val_pred, val_label = [], []
        with torch.no_grad():
            for x, y in loader["val_loader"].get_iterator():
                x, y = x.cuda(), y.cuda()
                n, p, pi = model(x)
                val_pred.append(zinb_mean(n, p, pi).cpu().numpy())
                val_label.append(inverse(scaler, y).cpu().numpy())
        val_pred, val_label = np.concatenate(val_pred), np.concatenate(val_label)
        np.savez(cache_path, pred=val_pred, label=val_label)

    # Ordinary split conformal.
    ordinary_q = q(np.abs(val_label - val_pred))
    ordinary = metrics(test_pred, np.maximum(test_pred - ordinary_q, 0), test_pred + ordinary_q, test_label)

    # Core ZeroCQR: disjoint tune/fit/cal splits, zero gate, residual centre,
    # then zero/positive prediction bins.  No auxiliary, periodic, or online
    # correction is included in this comparison.
    r_t, y_t = val_pred[::3], val_label[::3]
    r_f, y_f = val_pred[1::3], val_label[1::3]
    r_c, y_c = val_pred[2::3], val_label[2::3]
    vals = r_t[np.isfinite(r_t) & np.isfinite(y_t)]
    candidates = np.unique(np.quantile(vals, np.linspace(0, 0.95, 33)))
    base_mae = np.abs(np.maximum(r_f, 0) - y_f).mean()
    base_mse = ((np.maximum(r_f, 0) - y_f) ** 2).mean()
    best_gate, best_loss = candidates[0], np.inf
    for gate in candidates:
        active = (r_t > gate) & np.isfinite(r_t) & np.isfinite(y_t)
        shift_t = np.median((y_t - r_t)[active]) if active.any() else 0.0
        pred = np.where(r_f > gate, np.maximum(r_f + shift_t, 0), 0)
        loss = np.abs(pred - y_f).mean() / base_mae + ((pred - y_f) ** 2).mean() / base_mse
        if loss < best_loss:
            best_gate, best_loss = gate, loss
    active_f = (r_f > best_gate) & np.isfinite(r_f) & np.isfinite(y_f)
    shift = np.median((y_f - r_f)[active_f]) if active_f.any() else 0.0
    active_values = r_t[(r_t > best_gate) & np.isfinite(r_t) & np.isfinite(y_t)]
    edges = np.quantile(active_values, np.arange(1, bins) / bins)
    pred_c = np.where(r_c > best_gate, np.maximum(r_c + shift, 0), 0)
    scores = np.abs(y_c - pred_c)
    q0 = q(scores[r_c <= best_gate])
    qa = q(scores[r_c > best_gate])
    bins_c = np.searchsorted(edges, r_c, side="left")
    qb = np.array([q(scores[(r_c > best_gate) & (bins_c == i)]) for i in range(bins)])
    qb = np.where(np.isfinite(qb), qb, qa)
    point = np.where(test_pred > best_gate, np.maximum(test_pred + shift, 0), 0)
    point[point <= 1e-3] = 0
    bins_t = np.searchsorted(edges, test_pred, side="left")
    radius = np.where(test_pred <= best_gate, q0, qb[np.clip(bins_t, 0, bins - 1)])
    zero = metrics(point, np.maximum(point - radius, 0), point + radius, test_label)

    result = {"target_coverage": 95.0, "ordinary_od_cqr": ordinary, f"zero_cqr_{bins}bin_core": zero,
              "ordinary_radius": float(ordinary_q), "zero_gate": float(best_gate), "zero_shift": float(shift)}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"chi_pdr_no_context_mpiw_compare_{bins}bin.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(path)


if __name__ == "__main__":
    main()
