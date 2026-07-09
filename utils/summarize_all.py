"""Summarize all multi-dataset, multi-horizon flow results into result/res.txt.

Walks result/2025_12to{H}/<MODEL>/<dataset>/<ts>.log, parses the final
"Average: MAE .., MAPE .., RMSE .." line (and per-horizon lines), and writes
a tidy long-format table plus per-(dataset,horizon) leaderboards.
"""

import os
import re
import glob
from collections import defaultdict

RESULT_ROOT = "/home/dy23a.fsu/st/result"
HORIZONS = [1] #, 3, 6, 9, 12
DATASETS = ["nyc_manhattan_od_15min_taxi","nyc_manhattan_od_15min_bike","nyc_manhattan_od_15min_fhv",
             "chicago_od_15min_taxi", "chicago_od_15min_bike", "chicago_od_15min_tnp", 
             "dc_od_60min_taxi", "dc_od_60min_bike", 
             "sf_od_15min_taxi", "sf_od_15min_bike"]
OUT = os.path.join(RESULT_ROOT, "res.txt")

AVG_RE = re.compile(r"Average:\s*MAE:\s*([\d.]+),\s*MAPE:\s*([\d.]+),\s*RMSE:\s*([\d.]+)")
HOR_RE = re.compile(r"Test Horizon:\s*(\d+),\s*MAE:\s*([\d.]+),\s*MAPE:\s*([\d.]+),\s*RMSE:\s*([\d.]+)")


def parse_log(path):
    """Return (avg dict, {h: mae}) from the last matching lines, or (None, {})."""
    avg = None
    per_h = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = AVG_RE.search(line)
            if m:
                avg = dict(MAE=float(m.group(1)), MAPE=float(m.group(2)), RMSE=float(m.group(3)))
            mh = HOR_RE.search(line)
            if mh:
                per_h[int(mh.group(1))] = float(mh.group(2))
    return avg, per_h


def latest_log(model_dir, dataset):
    logs = glob.glob(os.path.join(model_dir, dataset, "*.log"))
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def main():
    # records[(dataset, horizon)][model] = (avg, per_h)
    records = defaultdict(dict)
    for H in HORIZONS:
        root = os.path.join(RESULT_ROOT, f"2025_12to{H}")
        if not os.path.isdir(root):
            continue
        for model in sorted(os.listdir(root)):
            mdir = os.path.join(root, model)
            if not os.path.isdir(mdir):
                continue
            for ds in DATASETS:
                lg = latest_log(mdir, ds)
                if not lg:
                    continue
                avg, per_h = parse_log(lg)
                if avg is None:
                    continue
                records[(ds, H)][model] = (avg, per_h)

    def is_stllm(m):
        return m.upper().startswith("STLLM")

    lines = []
    w = lines.append
    w("=" * 90)
    w("  MULTI-DATASET x MULTI-HORIZON FLOW RESULTS")
    w("  datasets: " + ", ".join(DATASETS))
    w("  horizons: 12-to-" + "/".join(str(h) for h in HORIZONS))
    w("=" * 90)

    # ---- Per (dataset, horizon) leaderboard sorted by MAE ----
    for ds in DATASETS:
        w("")
        w("#" * 90)
        w(f"# DATASET: {ds}")
        w("#" * 90)
        for H in HORIZONS:
            tab = records.get((ds, H))
            if not tab:
                continue
            w("")
            w(f"--- horizon 12->{H}  ({len(tab)} models) ---")
            w(f"{'Model':<14}{'MAE':>9}{'MAPE':>9}{'RMSE':>9}")
            ranked = sorted(tab.items(), key=lambda kv: kv[1][0]["MAE"])
            for model, (avg, _) in ranked:
                w(f"{model:<14}{avg['MAE']:>9.3f}{avg['MAPE']:>9.3f}{avg['RMSE']:>9.3f}")

    # ---- STLLM best vs baseline best, per (dataset, horizon) ----
    w("")
    w("=" * 90)
    w("  STLLM (best) vs BASELINE (best, excl. ST-LLM-plus) — MAE lead %")
    w("=" * 90)
    w(f"{'Dataset':<22}{'H':>4}{'bestSTLLM':>18}{'MAE':>8}{'bestBase':>14}{'MAE':>8}{'lead%':>8}")
    for ds in DATASETS:
        for H in HORIZONS:
            tab = records.get((ds, H))
            if not tab:
                continue
            stllm = {m: v[0]["MAE"] for m, v in tab.items() if is_stllm(m)}
            base = {m: v[0]["MAE"] for m, v in tab.items()
                    if not is_stllm(m) and m.upper() != "ST-LLM-PLUS"}
            if not stllm or not base:
                continue
            sm = min(stllm, key=stllm.get)
            bm = min(base, key=base.get)
            lead = (base[bm] - stllm[sm]) / base[bm] * 100
            w(f"{ds:<22}{H:>4}{sm:>18}{stllm[sm]:>8.3f}{bm:>14}{base[bm]:>8.3f}{lead:>+7.2f}%")

    # ---- Lead vs horizon, excluding TrustEnergy (the strongest multi-step baseline) ----
    w("")
    w("=" * 90)
    w("  STLLM (best) lead % over baselines — WITH vs WITHOUT TrustEnergy")
    w("=" * 90)
    w(f"{'Dataset':<22}{'H':>4}{'lead_all%':>11}{'lead_noTrust%':>15}")
    for ds in DATASETS:
        for H in HORIZONS:
            tab = records.get((ds, H))
            if not tab:
                continue
            stllm = {m: v[0]["MAE"] for m, v in tab.items() if is_stllm(m)}
            base = {m: v[0]["MAE"] for m, v in tab.items()
                    if not is_stllm(m) and m.upper() != "ST-LLM-PLUS"}
            if not stllm or not base:
                continue
            sbest = min(stllm.values())
            ball = min(base.values())
            base_nt = {m: v for m, v in base.items() if m.upper() != "TRUSTENERGY"}
            bnt = min(base_nt.values()) if base_nt else ball
            la = (ball - sbest) / ball * 100
            lnt = (bnt - sbest) / bnt * 100
            w(f"{ds:<22}{H:>4}{la:>+10.2f}%{lnt:>+14.2f}%")

    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print(text)
    print(f"\n[written to {OUT}]")


if __name__ == "__main__":
    main()
