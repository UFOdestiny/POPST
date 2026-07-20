"""Read-only ordinary-CQR and core-ZeroCQR comparison for Chicago PDR_REG."""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.od.pdr_reg.pdr_reg_model import PDRReg
from utils.dataloader import get_dataset_info, load_adj_from_numpy, load_dataset
from utils.graph_algo import normalize_adj_mx

OUT = os.path.join(ROOT, "result/Chi_OD_CALIBRATION_COMPARE")
SOURCE = os.path.join(ROOT, "result/Chi_OD/PDR_REG/chicago_od_15min_taxi")
CHECKPOINT = os.path.join(SOURCE, "PDR_REG_2026-07-17_09-10-34.pt")
ALPHA = 0.05


class Args:
    years = "2025_12to1"; seq_len = 12; horizon = 1; bs = 64


def q(x, level=1 - ALPHA):
    x = np.asarray(x)[np.isfinite(x)].reshape(-1)
    if not x.size: return 0.0
    rank = min(max(int(np.ceil((x.size + 1) * level)), 1), x.size) - 1
    return float(np.partition(x, rank)[rank])


def metric(point, radius, label):
    lower, upper = np.maximum(point - radius, 0), point + radius
    width = upper - lower
    score = width + 2 / ALPHA * np.maximum(lower-label, 0) + 2 / ALPHA * np.maximum(label-upper, 0)
    return {"MAE": float(np.abs(point-label).mean()), "MSE": float(((point-label)**2).mean()),
            "MPIW": float(width.mean()), "IS": float(score.mean()),
            "COV": float(((label >= lower) & (label <= upper)).mean() * 100)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--bins", type=int, default=8); ns = parser.parse_args()
    bins = max(ns.bins, 1)
    raw_test = np.load(os.path.join(SOURCE, "PDR_REG-chicago_od_15min_taxi-res.npy"), mmap_mode="r")
    test_raw, test_y = np.asarray(raw_test[0]), np.asarray(raw_test[1])
    cache_path = os.path.join(OUT, "chi_pdr_reg_validation_predictions.npz")
    if os.path.exists(cache_path):
        cache = np.load(cache_path); val_raw, val_y = cache["pred"], cache["label"]
    else:
        path, adj_path, n = get_dataset_info("chicago_od_15min_taxi", "2025_12to1")
        adj = load_adj_from_numpy(adj_path) - np.eye(n)
        model = PDRReg(A=normalize_adj_mx(adj, "uqgnn")[0], node_num=n, input_dim=n, output_dim=n,
                       seq_len=12, horizon=1, context_dim=64, zone_embed_dim=16, num_spatial_layers=2,
                       num_experts=3, head_hidden_dim=128, dropout=0).cuda().eval()
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=False))
        loaders, scaler = load_dataset(path, Args(), logging.getLogger("pdr_reg_compare"))
        ps, ys = [], []
        with torch.no_grad():
            for x, y in loaders["val_loader"].get_iterator():
                p = model(x.cuda())
                p = scaler.inverse_transform(p, device="cuda")
                y = scaler.inverse_transform(y.cuda(), device="cuda")
                ps.append(p.cpu().numpy()); ys.append(y.cpu().numpy())
        val_raw, val_y = np.concatenate(ps), np.concatenate(ys)
        np.savez(cache_path, pred=val_raw, label=val_y)

    ordinary_q = q(np.abs(val_y-val_raw)); ordinary = metric(test_raw, ordinary_q, test_y)
    rt, yt = val_raw[::3], val_y[::3]; rf, yf = val_raw[1::3], val_y[1::3]; rc, yc = val_raw[2::3], val_y[2::3]
    candidates = np.unique(np.quantile(rt[np.isfinite(rt)&np.isfinite(yt)], np.linspace(0, .95, 33)))
    base_mae, base_mse = np.abs(np.maximum(rf,0)-yf).mean(), ((np.maximum(rf,0)-yf)**2).mean()
    gate, best = candidates[0], np.inf
    for g in candidates:
        mask = (rt>g)&np.isfinite(rt)&np.isfinite(yt); s = np.median((yt-rt)[mask]) if mask.any() else 0
        p = np.where(rf>g, np.maximum(rf+s,0),0); loss=np.abs(p-yf).mean()/base_mae+((p-yf)**2).mean()/base_mse
        if loss<best: gate,best=g,loss
    m=(rf>gate)&np.isfinite(rf)&np.isfinite(yf); shift=np.median((yf-rf)[m]) if m.any() else 0
    pc=np.where(rc>gate,np.maximum(rc+shift,0),0); pc[pc<=1e-3]=0; scores=np.abs(yc-pc)
    edges=np.quantile(rt[(rt>gate)&np.isfinite(rt)&np.isfinite(yt)],np.arange(1,bins)/bins)
    ids=np.searchsorted(edges,rc,side="left"); active=rc>gate
    q0=q(scores[~active]); qa=q(scores[active]); qb=np.array([q(scores[active&(ids==i)]) for i in range(bins)])
    qb=np.where(np.isfinite(qb),qb,qa)
    point=np.where(test_raw>gate,np.maximum(test_raw+shift,0),0); point[point<=1e-3]=0
    test_ids=np.searchsorted(edges,test_raw,side="left"); radius=np.where(test_raw<=gate,q0,qb[np.clip(test_ids,0,bins-1)])
    result={"target_coverage":95,"ordinary_od_cqr":ordinary,f"zero_cqr_{bins}bin_core":metric(point,radius,test_y),"gate":float(gate),"shift":float(shift)}
    out=os.path.join(OUT,f"chi_pdr_reg_mpiw_compare_{bins}bin.json"); json.dump(result,open(out,"w"),indent=2)
    print(json.dumps(result,indent=2)); print(out)


if __name__ == "__main__": main()
