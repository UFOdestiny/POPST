"""Read-only ZeroCQR width-budget experiments for Chicago PDR_no_context.

The standard per-group 95% calibration is conservative when a huge predicted
zero group is already covered exactly.  This script allocates the *global*
95% coverage budget across zero and active OD cells, and selects all widths
using the held-out calibration third of validation only.
"""

import json
import os

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "result/Chi_OD_CALIBRATION_COMPARE")
SOURCE = os.path.join(ROOT, "result/Chi_OD/PDR_no_context/chicago_od_15min_taxi")
ALPHA = 0.05
TARGET = 1.0 - ALPHA


def q(scores, level):
    scores = np.asarray(scores)[np.isfinite(scores)].reshape(-1)
    if not scores.size:
        return 0.0
    rank = min(max(int(np.ceil((scores.size + 1) * level)), 1), scores.size) - 1
    return float(np.partition(scores, rank)[rank])


def report(point, radius, label):
    lower, upper = np.maximum(point - radius, 0), point + radius
    width = upper - lower
    err = point - label
    iscore = width + (2 / ALPHA) * np.maximum(lower - label, 0) + (2 / ALPHA) * np.maximum(label - upper, 0)
    return {
        "MAE": float(np.abs(err).mean()), "MSE": float((err ** 2).mean()),
        "MPIW": float(width.mean()), "IS": float(iscore.mean()),
        "COV": float(((label >= lower) & (label <= upper)).mean() * 100),
    }


def active_report(point, radius, label, active):
    lower, upper = np.maximum(point - radius, 0), point + radius
    mask = active.astype(bool)
    return {
        "MPIW": float((upper - lower)[mask].mean()),
        "COV": float(((label[mask] >= lower[mask]) & (label[mask] <= upper[mask])).mean() * 100),
        "fraction": float(mask.mean() * 100),
    }


def fit_gate(raw, label):
    tune_r, tune_y = raw[::3], label[::3]
    fit_r, fit_y = raw[1::3], label[1::3]
    values = tune_r[np.isfinite(tune_r) & np.isfinite(tune_y)]
    candidates = np.unique(np.quantile(values, np.linspace(0, 0.95, 33)))
    base = np.maximum(fit_r, 0)
    base_mae, base_mse = np.abs(base - fit_y).mean(), ((base - fit_y) ** 2).mean()
    best_gate, best_loss = candidates[0], np.inf
    for gate in candidates:
        active = (tune_r > gate) & np.isfinite(tune_r) & np.isfinite(tune_y)
        shift = np.median((tune_y - tune_r)[active]) if active.any() else 0.0
        point = np.where(fit_r > gate, np.maximum(fit_r + shift, 0), 0)
        loss = np.abs(point - fit_y).mean() / base_mae + ((point - fit_y) ** 2).mean() / base_mse
        if loss < best_loss:
            best_gate, best_loss = gate, loss
    active = (fit_r > best_gate) & np.isfinite(fit_r) & np.isfinite(fit_y)
    shift = np.median((fit_y - fit_r)[active]) if active.any() else 0.0
    return float(best_gate), float(shift)


def main():
    cache = np.load(os.path.join(OUT, "chi_pdr_no_context_validation_predictions.npz"))
    val_raw, val_y = cache["pred"], cache["label"]
    source = np.load(os.path.join(SOURCE, "PDR_no_context-chicago_od_15min_taxi-res.npy"), mmap_mode="r")
    test_raw, test_y = np.asarray(source[0]), np.asarray(source[1])
    gate, shift = fit_gate(val_raw, val_y)
    cal_raw, cal_y = val_raw[2::3], val_y[2::3]
    cal_point = np.where(cal_raw > gate, np.maximum(cal_raw + shift, 0), 0)
    cal_point[cal_point <= 1e-3] = 0
    test_point = np.where(test_raw > gate, np.maximum(test_raw + shift, 0), 0)
    test_point[test_point <= 1e-3] = 0
    zero_cal, active_cal = cal_raw <= gate, cal_raw > gate
    zero_covered = ((cal_y == 0) & zero_cal).sum()
    active_count = active_cal.sum()
    # Minimal active conditional coverage needed to reach 95% globally on
    # calibration, after exact predicted-zero cells have consumed coverage.
    active_level = np.clip((TARGET * cal_y.size - zero_covered) / max(active_count, 1), 0, 1)
    active_scores = np.abs(cal_y - cal_point)[active_cal]
    active_q = q(active_scores, active_level)
    radius_budget = np.where(test_raw <= gate, 0.0, active_q)

    # Standard 8-bin radii, then select the smallest common scale whose
    # calibration coverage reaches 95%.  This preserves relative bin widths.
    edges = np.quantile(cal_raw[active_cal], np.arange(1, 8) / 8)
    cal_bins = np.searchsorted(edges, cal_raw, side="left")
    bin_q = np.array([q(np.abs(cal_y - cal_point)[active_cal & (cal_bins == b)], TARGET) for b in range(8)])
    cal_radius = np.where(zero_cal, 0.0, bin_q[np.clip(cal_bins, 0, 7)])
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        cov = ((cal_y >= np.maximum(cal_point - mid * cal_radius, 0)) & (cal_y <= cal_point + mid * cal_radius)).mean()
        if cov >= TARGET:
            hi = mid
        else:
            lo = mid
    test_bins = np.searchsorted(edges, test_raw, side="left")
    radius_scaled = np.where(test_raw <= gate, 0.0, hi * bin_q[np.clip(test_bins, 0, 7)])
    test_active = test_raw > gate
    active_constraints = {}
    for level in (0.50, 0.80, 0.90, 0.95):
        radius = np.where(test_active, q(active_scores, level), 0.0)
        active_constraints[f"active_{int(level * 100)}"] = {
            "all_cells": report(test_point, radius, test_y),
            "active_cells": active_report(test_point, radius, test_y, test_active),
        }

    result = {
        "target_coverage": 95.0, "gate": gate, "shift": shift,
        "zero_point_no_interval": report(test_point, np.zeros_like(test_point), test_y),
        "budgeted_active_common_radius": report(test_point, radius_budget, test_y),
        "budgeted_active_common_q": active_q, "active_required_coverage": float(active_level * 100),
        "scaled_8bin": report(test_point, radius_scaled, test_y),
        "scaled_8bin_scale": float(hi),
        "active_coverage_constraints": active_constraints,
    }
    path = os.path.join(OUT, "chi_pdr_no_context_zero_cqr_budget_exploration.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(path)


if __name__ == "__main__":
    main()
