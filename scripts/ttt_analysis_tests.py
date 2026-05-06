#!/usr/bin/env python3
"""TTT analysis: error distribution, outliers, batch vs online, init sensitivity."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))
RECORDS_PATH = REPO_ROOT / "src" / "sim" / "data" / "encounter" / "records.json"

DEFAULT_WEIGHTS = {"priority": 0.20, "geometry": 0.34, "duration": 0.18,
                   "imagery": 0.16, "clarity": 0.12}
_REALIZED = {"accept": 0.85, "refine": 0.55}
KEYS = list(DEFAULT_WEIGHTS)

def _load():
    rows = []
    for r in json.loads(RECORDS_PATH.read_text())["records"]:
        d = r.get("decision", {})
        td = d.get("trust_details", {})
        act = d.get("observation_recommended_action")
        if td and act in _REALIZED:
            sc = r.get("features", {}).get("score_components", {})
            rows.append({
                "features": {
                    "priority": td.get("target_priority", sc.get("priority", td.get("agreement", 0.5))),
                    "geometry": td.get("geometry_margin", 0.5),
                    "duration": min(td.get("duration_margin", 0.5), 0.7),
                    "imagery":  min(td.get("imagery_support", 0.5), 0.7),
                    "clarity":  td.get("clarity_support", 0.5),
                },
                "util": _REALIZED[act], "action": act,
            })
    return rows

def _predict(w, f): return sum(w[k]*f[k] for k in w)

def _step(weights, features, util, lr, reg=0.002):
    pred = _predict(weights, features)
    err = util - pred
    for k in list(weights):
        weights[k] += lr * err * features[k] + reg * (DEFAULT_WEIGHTS[k] - weights[k])
        weights[k] = max(0.001, weights[k])
    s = sum(weights.values())
    return {k: v/s for k, v in weights.items()}, err

def _run(traces, cycles=20, lr=0.02, seed=42, w0=None):
    rng = np.random.default_rng(seed)
    w = dict(w0 if w0 else DEFAULT_WEIGHTS)
    for _ in range(cycles):
        for i in rng.permutation(len(traces)):
            w, _ = _step(w, traces[i]["features"], traces[i]["util"], lr)
    return w

def _mae(w, traces):
    return float(np.mean([abs(t["util"] - _predict(w, t["features"])) for t in traces]))

ENC = _load()
KEYS = list(DEFAULT_WEIGHTS)

# ── 1. ERROR DISTRIBUTION ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. ERROR DISTRIBUTION (prior vs learned weights)")
print("="*60)
w_prior   = dict(DEFAULT_WEIGHTS)
w_learned = _run(ENC)

for label, w in [("Prior", w_prior), ("Learned", w_learned)]:
    errs = [t["util"] - _predict(w, t["features"]) for t in ENC]
    ae   = [abs(e) for e in errs]
    print(f"\n  {label} weights:")
    print(f"    Mean error (bias):  {np.mean(errs):+.5f}")
    print(f"    MAE:                {np.mean(ae):.5f}")
    print(f"    Median AE:          {np.median(ae):.5f}")
    print(f"    p75 AE:             {np.percentile(ae, 75):.5f}")
    print(f"    p90 AE:             {np.percentile(ae, 90):.5f}")
    print(f"    p99 AE:             {np.percentile(ae, 99):.5f}")
    print(f"    Max AE:             {np.max(ae):.5f}")
    print(f"    Std AE:             {np.std(ae):.5f}")
    # histogram
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]
    counts, _ = np.histogram(ae, bins=bins)
    print(f"    Distribution:")
    for i, c in enumerate(counts):
        bar = "#" * (c * 30 // max(counts, default=1))
        print(f"      [{bins[i]:.2f}-{bins[i+1]:.2f}]: {c:3d}  {bar}")


# ── 2. OUTLIER ANALYSIS ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. OUTLIER ANALYSIS (top-10 highest-error traces after TTT)")
print("="*60)
w = _run(ENC)
scored = [(abs(t["util"] - _predict(w, t["features"])), t) for t in ENC]
scored.sort(reverse=True)
print(f"\n  {'Rank':>4}  {'AE':>7}  {'action':>8}  {'util':>6}  {'pred':>6}  "
      f"{'clarity':>8}  {'geometry':>9}  {'priority':>9}")
print("  " + "-"*72)
for rank, (ae, t) in enumerate(scored[:10], 1):
    pred = _predict(w, t["features"])
    f = t["features"]
    print(f"  {rank:>4}  {ae:>7.4f}  {t['action']:>8}  {t['util']:>6.2f}  {pred:>6.3f}  "
          f"  {f['clarity']:>6.3f}  {f['geometry']:>9.3f}  {f['priority']:>9.3f}")

# Common pattern in outliers?
top10_f = {k: np.mean([t["features"][k] for _, t in scored[:10]]) for k in KEYS}
all_f   = {k: np.mean([t["features"][k] for t in ENC]) for k in KEYS}
print(f"\n  Feature averages — top-10 outliers vs full corpus:")
print(f"  {'Feature':<12}  {'Outliers':>9}  {'Corpus':>8}  {'Delta':>8}")
for k in KEYS:
    delta = top10_f[k] - all_f[k]
    print(f"  {k:<12}  {top10_f[k]:>9.4f}  {all_f[k]:>8.4f}  {delta:>+8.4f}")


# ── 3. BATCH vs ONLINE ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. BATCH vs ONLINE (least-squares optimal vs TTT online)")
print("="*60)
# Batch: solve w* = argmin ||Xw - y||^2 with simplex constraint (normalized)
# Use unconstrained OLS then normalize
X = np.array([[t["features"][k] for k in KEYS] for t in ENC])
y = np.array([t["util"] for t in ENC])
# OLS
w_ols, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
# Clip negatives and normalize to get valid weight vector
w_ols_clipped = np.maximum(w_ols, 0.001)
w_ols_norm = w_ols_clipped / w_ols_clipped.sum()
w_batch = dict(zip(KEYS, w_ols_norm))
w_online = _run(ENC)

mae_prior  = _mae(DEFAULT_WEIGHTS, ENC)
mae_batch  = _mae(w_batch, ENC)
mae_online = _mae(w_online, ENC)
# Unconstrained OLS MAE (raw, not simplex-constrained)
w_raw = dict(zip(KEYS, w_ols))
pred_raw = X @ w_ols
mae_raw = float(np.mean(np.abs(pred_raw - y)))

print(f"\n  {'Method':<22}  {'MAE':>8}  {'Improvement':>12}")
print("  " + "-"*46)
print(f"  {'Prior (default)':<22}  {mae_prior:>8.5f}  {'baseline':>12}")
print(f"  {'TTT online (lr=0.02)':<22}  {mae_online:>8.5f}  {(mae_prior-mae_online)/mae_prior*100:>11.1f}%")
print(f"  {'Batch OLS (simplex)':<22}  {mae_batch:>8.5f}  {(mae_prior-mae_batch)/mae_prior*100:>11.1f}%")
print(f"  {'OLS unconstrained':<22}  {mae_raw:>8.5f}  {(mae_prior-mae_raw)/mae_prior*100:>11.1f}%")

print(f"\n  Batch OLS weights (simplex-constrained):")
for k in KEYS:
    print(f"    {k:<10}  prior={DEFAULT_WEIGHTS[k]:.4f}  batch={w_batch[k]:.4f}  "
          f"online={w_online[k]:.4f}  {'AGREE' if abs(w_batch[k]-w_online[k]) < 0.05 else 'DIVERGE'}")
efficiency = (mae_prior - mae_online) / (mae_prior - mae_batch) * 100 if mae_prior > mae_batch else 0
print(f"\n  TTT captures {efficiency:.1f}% of the batch-optimal improvement")


# ── 4. INITIALIZATION SENSITIVITY ─────────────────────────────────────────────
print("\n" + "="*60)
print("4. INITIALIZATION SENSITIVITY (different starting weights)")
print("="*60)
inits = {
    "default":     {"priority": 0.20, "geometry": 0.34, "duration": 0.18, "imagery": 0.16, "clarity": 0.12},
    "uniform":     {k: 0.20 for k in KEYS},
    "clarity-hot": {"priority": 0.10, "geometry": 0.10, "duration": 0.10, "imagery": 0.10, "clarity": 0.60},
    "geom-hot":    {"priority": 0.05, "geometry": 0.70, "duration": 0.05, "imagery": 0.10, "clarity": 0.10},
    "random-1":    dict(zip(KEYS, [0.31, 0.18, 0.22, 0.14, 0.15])),
    "random-2":    dict(zip(KEYS, [0.15, 0.25, 0.15, 0.30, 0.15])),
}
print(f"\n  {'Init':<14}  {'Init MAE':>9}  {'Final MAE':>10}  {'Improv':>8}  {'clarity':>8}  {'geometry':>9}")
print("  " + "-"*64)
for name, w0 in inits.items():
    # normalize
    s = sum(w0.values()); w0n = {k: v/s for k, v in w0.items()}
    init_mae = _mae(w0n, ENC)
    w_final = _run(ENC, w0=w0n)
    final_mae = _mae(w_final, ENC)
    pct = (init_mae - final_mae) / init_mae * 100
    print(f"  {name:<14}  {init_mae:>9.5f}  {final_mae:>10.5f}  {pct:>7.1f}%  "
          f"{w_final['clarity']:>8.4f}  {w_final['geometry']:>9.4f}")


# ── 5. WEIGHT TRAJECTORY (per-update, first 256 updates) ──────────────────────
print("\n" + "="*60)
print("5. WEIGHT TRAJECTORY — per-update clarity & geometry (first pass)")
print("   Shows how quickly individual weights move on first encounter")
print("="*60)
rng = np.random.default_rng(42)
w = dict(DEFAULT_WEIGHTS)
order = rng.permutation(len(ENC))
checkpoints = set([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 256])
print(f"\n  {'Update':>7}  {'clarity':>8}  {'geometry':>9}  {'priority':>9}  {'MAE':>8}")
print("  " + "-"*46)
for n, idx in enumerate(order, 1):
    w, _ = _step(w, ENC[idx]["features"], ENC[idx]["util"], 0.02)
    if n in checkpoints:
        mae = _mae(w, ENC)
        print(f"  {n:>7}  {w['clarity']:>8.4f}  {w['geometry']:>9.4f}  "
              f"{w['priority']:>9.4f}  {mae:>8.5f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
