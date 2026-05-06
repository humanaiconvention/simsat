#!/usr/bin/env python3
"""Final TTT tests: k-fold CV, threshold calibration, weight perturbation stability."""
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


# ── 1. K-FOLD CROSS-VALIDATION (holdout generalization) ───────────────────────
print("\n" + "="*60)
print("1. 5-FOLD CROSS-VALIDATION (holdout generalization)")
print("="*60)
k = 5
rng = np.random.default_rng(42)
indices = rng.permutation(len(ENC))
folds = np.array_split(indices, k)

prior_maes, trained_maes = [], []
print(f"\n  {'Fold':>5}  {'Train N':>8}  {'Test N':>7}  {'Prior MAE':>10}  {'TTT MAE':>9}  {'Improv':>8}")
print("  " + "-"*54)
for fold_i in range(k):
    test_idx  = folds[fold_i]
    train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_i])
    train = [ENC[i] for i in train_idx]
    test  = [ENC[i] for i in test_idx]
    w = _run(train, cycles=20)
    prior_mae = _mae(DEFAULT_WEIGHTS, test)
    ttt_mae   = _mae(w, test)
    pct = (prior_mae - ttt_mae) / prior_mae * 100
    prior_maes.append(prior_mae); trained_maes.append(ttt_mae)
    print(f"  {fold_i+1:>5}  {len(train):>8}  {len(test):>7}  {prior_mae:>10.5f}  "
          f"{ttt_mae:>9.5f}  {pct:>7.1f}%")
print(f"\n  Mean: prior MAE {np.mean(prior_maes):.5f}  TTT MAE {np.mean(trained_maes):.5f}  "
      f"({(np.mean(prior_maes)-np.mean(trained_maes))/np.mean(prior_maes)*100:.1f}%)")
print(f"  Std:  prior MAE {np.std(prior_maes):.5f}  TTT MAE {np.std(trained_maes):.5f}")


# ── 2. THRESHOLD CALIBRATION (accept vs refine discrimination) ─────────────────
print("\n" + "="*60)
print("2. THRESHOLD CALIBRATION")
print("   Can the learned score discriminate accept from refine?")
print("="*60)
w_prior   = dict(DEFAULT_WEIGHTS)
w_learned = _run(ENC)

for label, w in [("Prior", w_prior), ("Learned", w_learned)]:
    accept_preds = [_predict(w, t["features"]) for t in ENC if t["action"] == "accept"]
    refine_preds = [_predict(w, t["features"]) for t in ENC if t["action"] == "refine"]
    print(f"\n  {label}:  accept pred {np.mean(accept_preds):.4f}±{np.std(accept_preds):.4f}"
          f"  refine pred {np.mean(refine_preds):.4f}±{np.std(refine_preds):.4f}")
    # Sweep thresholds
    all_preds = [(p, "accept") for p in accept_preds] + [(p, "refine") for p in refine_preds]
    thresholds = np.linspace(0.5, 0.8, 13)
    best_acc, best_thr = 0, 0
    for thr in thresholds:
        correct = sum(1 for p, a in all_preds
                      if (p >= thr and a == "accept") or (p < thr and a == "refine"))
        acc = correct / len(all_preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    # Overlap
    overlap = sum(1 for p in accept_preds if p < np.mean(refine_preds) + np.std(refine_preds))
    print(f"  Best threshold: {best_thr:.2f}  accuracy: {best_acc:.3f}  "
          f"(accept overlap with refine dist: {overlap}/{len(accept_preds)})")


# ── 3. WEIGHT PERTURBATION STABILITY ──────────────────────────────────────────
print("\n" + "="*60)
print("3. WEIGHT PERTURBATION STABILITY")
print("   Perturb learned weights, re-run TTT — how fast does it recover?")
print("="*60)
w_star = _run(ENC, cycles=50)  # ground truth attractor
mae_star = _mae(w_star, ENC)

perturbations = {
    "none (ground truth)":  dict(w_star),
    "+0.05 geometry":       {k: (v + (0.05 if k == "geometry" else 0)) for k, v in w_star.items()},
    "+0.10 geometry":       {k: (v + (0.10 if k == "geometry" else 0)) for k, v in w_star.items()},
    "-0.05 clarity":        {k: (v + (-0.05 if k == "clarity" else 0)) for k, v in w_star.items()},
    "add noise σ=0.05":     {k: v + np.random.default_rng(7).normal(0, 0.05) for k, v in w_star.items()},
    "back to prior":        dict(DEFAULT_WEIGHTS),
}
print(f"\n  {'Perturbation':<26}  {'Init MAE':>9}  {'1-cycle':>8}  {'5-cycle':>8}  {'Recovery':>9}")
print("  " + "-"*60)
for name, w0 in perturbations.items():
    s = sum(max(v, 0.001) for v in w0.values())
    w0n = {k: max(v, 0.001)/s for k, v in w0.items()}
    init_mae = _mae(w0n, ENC)
    w1 = _run(ENC, cycles=1, w0=w0n)
    w5 = _run(ENC, cycles=5, w0=w0n)
    mae1 = _mae(w1, ENC)
    mae5 = _mae(w5, ENC)
    gap_init = init_mae - mae_star
    gap5 = mae5 - mae_star
    recovery = (1 - gap5 / gap_init) * 100 if abs(gap_init) > 1e-6 else 100
    print(f"  {name:<26}  {init_mae:>9.5f}  {mae1:>8.5f}  {mae5:>8.5f}  {recovery:>8.1f}%")


# ── 4. IRREDUCIBLE ERROR ANALYSIS ─────────────────────────────────────────────
print("\n" + "="*60)
print("4. IRREDUCIBLE ERROR ANALYSIS")
print("   What error floor remains regardless of weight choice?")
print("="*60)
w_learned = _run(ENC, cycles=100)
w_ols_raw = np.linalg.lstsq(
    np.array([[t["features"][k] for k in KEYS] for t in ENC]),
    np.array([t["util"] for t in ENC]), rcond=None)[0]
w_ols = dict(zip(KEYS, w_ols_raw))

# MAE per action
for label, w in [("TTT online", w_learned), ("OLS unconstrained", w_ols)]:
    acc_errs = [abs(t["util"] - _predict(w, t["features"])) for t in ENC if t["action"] == "accept"]
    ref_errs = [abs(t["util"] - _predict(w, t["features"])) for t in ENC if t["action"] == "refine"]
    print(f"\n  {label}:")
    print(f"    accept  MAE={np.mean(acc_errs):.5f}  median={np.median(acc_errs):.5f}  max={np.max(acc_errs):.5f}")
    print(f"    refine  MAE={np.mean(ref_errs):.5f}  median={np.median(ref_errs):.5f}  max={np.max(ref_errs):.5f}")

# R-squared (explained variance)
X = np.array([[t["features"][k] for k in KEYS] for t in ENC])
y = np.array([t["util"] for t in ENC])
for label, w in [("Prior", DEFAULT_WEIGHTS), ("TTT online", w_learned)]:
    preds = np.array([_predict(w, t["features"]) for t in ENC])
    ss_res = np.sum((y - preds)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n  R² ({label}): {r2:.4f}  (explains {r2*100:.1f}% of utility variance)")

print("\n" + "="*60)
print("DONE")
print("="*60)
