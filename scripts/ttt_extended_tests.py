#!/usr/bin/env python3
"""Extended TTT test battery: ablation, learning curve, cross-pack, noise, convergence speed."""
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

def _load():
    rows = []
    for r in json.loads(RECORDS_PATH.read_text())["records"]:
        d = r.get("decision", {})
        td = d.get("trust_details", {})
        ls = d.get("learned_score")
        act = d.get("observation_recommended_action")
        if td and ls is not None and act in _REALIZED:
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
                "pack": r.get("scenario_pack", "unknown"),
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

def _run(traces, cycles=20, lr=0.02, seed=42):
    rng = np.random.default_rng(seed)
    w = dict(DEFAULT_WEIGHTS)
    for _ in range(cycles):
        for i in rng.permutation(len(traces)):
            w, _ = _step(w, traces[i]["features"], traces[i]["util"], lr)
    return w

def _mae(w, traces):
    return float(np.mean([abs(t["util"] - _predict(w, t["features"])) for t in traces]))

TRACES = _load()
KEYS = list(DEFAULT_WEIGHTS)

# ── 1. FEATURE ABLATION ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. FEATURE ABLATION (remove one feature at a time)")
print("="*60)
base_w = _run(TRACES)
base_mae = _mae(base_w, TRACES)
print(f"  Baseline (all features): MAE {base_mae:.5f}")
print(f"  {'Ablated feature':<14}  {'MAE':>8}  {'Delta MAE':>10}  {'Impact'}")
print("  " + "-"*50)
for drop in KEYS:
    ablated = [{"features": {k: v for k, v in t["features"].items() if k != drop},
                "util": t["util"]} for t in TRACES]
    w_abl = {"priority": 0.20, "geometry": 0.34, "duration": 0.18,
             "imagery": 0.16, "clarity": 0.12}
    w_abl.pop(drop)
    s = sum(w_abl.values()); w_abl = {k: v/s for k,v in w_abl.items()}

    def _step_abl(weights, features, util, lr, reg=0.002, drop=drop):
        pred = sum(weights[k]*features[k] for k in weights)
        err = util - pred
        ref = {k: v for k,v in DEFAULT_WEIGHTS.items() if k != drop}
        s0 = sum(ref.values()); ref = {k: v/s0 for k,v in ref.items()}
        for k in list(weights):
            weights[k] += lr * err * features[k] + reg * (ref[k] - weights[k])
            weights[k] = max(0.001, weights[k])
        s = sum(weights.values())
        return {k: v/s for k, v in weights.items()}

    rng = np.random.default_rng(42)
    wa = {k: v for k, v in w_abl.items()}
    for _ in range(20):
        for i in rng.permutation(len(ablated)):
            wa = _step_abl(wa, ablated[i]["features"], ablated[i]["util"], 0.02)
    mae_abl = float(np.mean([abs(t["util"] - sum(wa[k]*t["features"][k] for k in wa))
                              for t in ablated]))
    delta = mae_abl - base_mae
    print(f"  {drop:<14}  {mae_abl:>8.5f}  {delta:>+10.5f}  {'*** high impact' if delta > 0.005 else ''}")

# ── 2. LEARNING CURVE ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. LEARNING CURVE (corpus size vs MAE improvement)")
print("="*60)
rng0 = np.random.default_rng(42)
all_idx = rng0.permutation(len(TRACES))
init_mae_full = _mae(dict(DEFAULT_WEIGHTS), TRACES)
print(f"  {'N traces':<10}  {'Init MAE':>9}  {'Final MAE':>10}  {'Improvement':>12}")
print("  " + "-"*46)
for n in [25, 50, 75, 100, 128, 192, 256]:
    subset = [TRACES[i] for i in all_idx[:n]]
    w = _run(subset, cycles=20)
    init = _mae(dict(DEFAULT_WEIGHTS), subset)
    final = _mae(w, subset)
    pct = (init - final) / init * 100 if init > 0 else 0
    print(f"  {n:<10}  {init:>9.5f}  {final:>10.5f}  {pct:>11.1f}%")

# ── 3. CROSS-PACK GENERALIZATION ──────────────────────────────────────────────
print("\n" + "="*60)
print("3. CROSS-PACK GENERALIZATION (train on 2 packs, eval on held-out)")
print("="*60)
packs = sorted(set(t["pack"] for t in TRACES))
print(f"  Packs: {packs}")
print(f"  {'Train packs':<42}  {'Held-out pack':<30}  {'Train MAE':>9}  {'Holdout MAE':>12}")
print("  " + "-"*100)
for held in packs:
    train = [t for t in TRACES if t["pack"] != held]
    test  = [t for t in TRACES if t["pack"] == held]
    if not train or not test: continue
    w = _run(train, cycles=20)
    train_mae = _mae(w, train)
    test_mae  = _mae(w, test)
    init_test = _mae(dict(DEFAULT_WEIGHTS), test)
    pct = (init_test - test_mae) / init_test * 100 if init_test > 0 else 0
    train_str = ", ".join(p for p in packs if p != held)
    print(f"  {train_str:<42}  {held:<30}  {train_mae:>9.5f}  {test_mae:>12.5f}  ({pct:+.1f}%)")

# ── 4. NOISE ROBUSTNESS ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. NOISE ROBUSTNESS (Gaussian feature noise)")
print("="*60)
print(f"  {'Noise sigma':<14}  {'Final MAE':>10}  {'Improvement':>12}  {'clarity':>8}  {'geometry':>9}")
print("  " + "-"*58)
init_mae = _mae(dict(DEFAULT_WEIGHTS), TRACES)
for sigma in [0.0, 0.02, 0.05, 0.10, 0.20]:
    rng = np.random.default_rng(42)
    w = dict(DEFAULT_WEIGHTS)
    for _ in range(20):
        for i in rng.permutation(len(TRACES)):
            t = TRACES[i]
            noisy = {k: float(np.clip(v + rng.normal(0, sigma), 0.001, 0.999))
                     for k, v in t["features"].items()}
            w, _ = _step(w, noisy, t["util"], 0.02)
    fmae = _mae(w, TRACES)
    pct = (init_mae - fmae) / init_mae * 100
    print(f"  {sigma:<14.2f}  {fmae:>10.5f}  {pct:>11.1f}%  {w['clarity']:>8.4f}  {w['geometry']:>9.4f}")

# ── 5. CONVERGENCE SPEED ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. CONVERGENCE SPEED (cycles to reach % of total improvement)")
print("="*60)
w_final = _run(TRACES, cycles=100)
mae_final = _mae(w_final, TRACES)
mae_init = _mae(dict(DEFAULT_WEIGHTS), TRACES)
total_improvement = mae_init - mae_final

rng = np.random.default_rng(42)
w = dict(DEFAULT_WEIGHTS)
milestones = {50: None, 75: None, 90: None, 95: None, 99: None}
print(f"  Total improvement pool: {total_improvement:.5f}  (init {mae_init:.5f} -> asymptote {mae_final:.5f})")
print(f"  {'Milestone':<12}  {'At cycle':>9}  {'MAE':>8}  {'clarity':>8}  {'geometry':>9}")
print("  " + "-"*52)
for cycle in range(1, 51):
    for i in rng.permutation(len(TRACES)):
        w, _ = _step(w, TRACES[i]["features"], TRACES[i]["util"], 0.02)
    mae_now = _mae(w, TRACES)
    pct_captured = (mae_init - mae_now) / total_improvement * 100 if total_improvement > 0 else 0
    for m in list(milestones):
        if milestones[m] is None and pct_captured >= m:
            milestones[m] = (cycle, mae_now, w['clarity'], w['geometry'])
    if all(v is not None for v in milestones.values()):
        break
for m, v in sorted(milestones.items()):
    if v:
        print(f"  {m}% of pool  {v[0]:>9d}  {v[1]:>8.5f}  {v[2]:>8.4f}  {v[3]:>9.4f}")

# ── 6. ACTION-STRATIFIED SIGNAL ───────────────────────────────────────────────
print("\n" + "="*60)
print("6. ACTION-STRATIFIED WEIGHT SIGNAL (accept vs refine)")
print("="*60)
for action in ("accept", "refine"):
    subset = [t for t in TRACES if t["action"] == action]
    w = _run(subset, cycles=20)
    mae = _mae(w, subset)
    drifts = {k: w[k] - DEFAULT_WEIGHTS[k] for k in KEYS}
    print(f"\n  action={action}  n={len(subset)}  MAE={mae:.5f}")
    for k in KEYS:
        bar = "+" * int(abs(drifts[k]) * 100) if drifts[k] > 0 else "-" * int(abs(drifts[k]) * 100)
        print(f"    {k:<10}  {DEFAULT_WEIGHTS[k]:.4f} -> {w[k]:.4f}  ({drifts[k]:+.4f})  {bar}")

print("\n" + "="*60)
print("DONE")
print("="*60)
