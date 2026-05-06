#!/usr/bin/env python3
"""Deeper TTT tests: feature interactions, gradient consistency, utility sensitivity,
long-run stability, clip sensitivity, streaming simulation, cross-dataset convergence."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))
RECORDS_PATH  = REPO_ROOT / "src" / "sim" / "data" / "encounter" / "records.json"
TRACES_PATH   = REPO_ROOT / "src" / "sim" / "data" / "observation_vla" / "traces.json"

DEFAULT_WEIGHTS = {"priority": 0.20, "geometry": 0.34, "duration": 0.18,
                   "imagery": 0.16, "clarity": 0.12}
_REALIZED = {"accept": 0.85, "refine": 0.55}
KEYS = list(DEFAULT_WEIGHTS)

def _load_enc():
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

def _load_ovl():
    rows = []
    for t in json.loads(TRACES_PATH.read_text())["traces"]:
        da = t.get("decision_after") or {}
        td = da.get("trust_details")
        ls = da.get("learned_score")
        action = (t.get("assessment") or {}).get("recommended_action")
        if td and ls is not None and action in _REALIZED:
            rows.append({
                "features": {
                    "priority": td.get("target_priority", td.get("agreement", 0.5)),
                    "geometry": td.get("geometry_margin", 0.5),
                    "duration": min(td.get("duration_margin", 0.5), 0.7),
                    "imagery":  min(td.get("imagery_support", 0.5), 0.7),
                    "clarity":  td.get("clarity_support", 0.5),
                },
                "util": _REALIZED[action], "action": action,
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

ENC = _load_enc()
OVL = _load_ovl()


# ── 1. FEATURE CORRELATION MATRIX ─────────────────────────────────────────────
print("\n" + "="*60)
print("1. FEATURE CORRELATION MATRIX (multicollinearity check)")
print("="*60)
feat_mat = np.array([[t["features"][k] for k in KEYS] for t in ENC])
utils    = np.array([t["util"] for t in ENC])
print(f"\n  Pearson r with utility:")
for i, k in enumerate(KEYS):
    r = float(np.corrcoef(feat_mat[:, i], utils)[0, 1])
    print(f"    {k:<10}  r={r:+.4f}")
print(f"\n  Inter-feature correlation matrix:")
print(f"  {'':12}", end="")
for k in KEYS: print(f"  {k[:7]:>7}", end="")
print()
corr = np.corrcoef(feat_mat.T)
for i, ki in enumerate(KEYS):
    print(f"  {ki:<12}", end="")
    for j, kj in enumerate(KEYS):
        v = corr[i, j]
        flag = " *" if abs(v) > 0.5 and i != j else ""
        print(f"  {v:>+7.3f}{flag[:1]}", end="")
    print()


# ── 2. GRADIENT DIRECTION CONSISTENCY ─────────────────────────────────────────
print("\n" + "="*60)
print("2. GRADIENT DIRECTION CONSISTENCY")
print("   (% of traces that push each weight in the final learned direction)")
print("="*60)
w_learned = _run(ENC)
gradients = {k: [] for k in KEYS}
for t in ENC:
    pred = _predict(w_learned, t["features"])
    err  = t["util"] - pred
    for k in KEYS:
        gradients[k].append(err * t["features"][k])

final_drift = {k: w_learned[k] - DEFAULT_WEIGHTS[k] for k in KEYS}
print(f"\n  {'Feature':<12}  {'Final drift':>12}  {'% agree':>8}  {'Mean grad':>10}  {'Grad std':>9}")
print("  " + "-"*56)
for k in KEYS:
    g = np.array(gradients[k])
    direction = np.sign(final_drift[k])
    pct_agree = float(np.mean(np.sign(g) == direction)) * 100
    print(f"  {k:<12}  {final_drift[k]:>+12.4f}  {pct_agree:>7.1f}%  "
          f"{np.mean(g):>+10.5f}  {np.std(g):>9.5f}")


# ── 3. UTILITY MAPPING SENSITIVITY ────────────────────────────────────────────
print("\n" + "="*60)
print("3. UTILITY MAPPING SENSITIVITY")
print("   (how much do the accept/refine utility values matter?)")
print("="*60)
print(f"\n  {'accept util':>12}  {'refine util':>12}  {'Final MAE':>10}  {'clarity':>8}  {'geometry':>9}")
print("  " + "-"*56)
for au, ru in [(0.70, 0.40), (0.80, 0.50), (0.85, 0.55), (0.90, 0.60), (1.00, 0.70)]:
    traces_u = [{"features": t["features"],
                 "util": au if t["action"] == "accept" else ru,
                 "action": t["action"]} for t in ENC]
    w = _run(traces_u)
    mae = _mae(w, traces_u)
    print(f"  {au:>12.2f}  {ru:>12.2f}  {mae:>10.5f}  {w['clarity']:>8.4f}  {w['geometry']:>9.4f}")


# ── 4. LONG-RUN STABILITY (500 cycles) ────────────────────────────────────────
print("\n" + "="*60)
print("4. LONG-RUN STABILITY (500 cycles — drift and oscillation check)")
print("="*60)
rng = np.random.default_rng(42)
w = dict(DEFAULT_WEIGHTS)
snapshots = {}
for cycle in range(1, 501):
    for i in rng.permutation(len(ENC)):
        w, _ = _step(w, ENC[i]["features"], ENC[i]["util"], 0.02)
    if cycle in (1, 5, 10, 20, 50, 100, 200, 500):
        snapshots[cycle] = (dict(w), _mae(w, ENC))

print(f"\n  {'Cycle':>6}  {'MAE':>8}", end="")
for k in KEYS: print(f"  {k[:6]:>7}", end="")
print()
print("  " + "-"*66)
for cycle, (ws, mae) in sorted(snapshots.items()):
    print(f"  {cycle:>6}  {mae:>8.5f}", end="")
    for k in KEYS: print(f"  {ws[k]:>7.4f}", end="")
    print()


# ── 5. CLIP SENSITIVITY ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. CLIP SENSITIVITY (duration/imagery saturation threshold)")
print("="*60)
print(f"\n  {'dur_clip':>8}  {'img_clip':>8}  {'MAE':>8}  {'clarity':>8}  {'geometry':>9}  {'duration':>9}  {'imagery':>8}")
print("  " + "-"*64)
for dur_clip, img_clip in [(0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (1.0, 1.0), (0.7, 1.0)]:
    traces_c = []
    for r in json.loads(RECORDS_PATH.read_text())["records"]:
        d = r.get("decision", {})
        td = d.get("trust_details", {})
        act = d.get("observation_recommended_action")
        if td and act in _REALIZED:
            sc = r.get("features", {}).get("score_components", {})
            traces_c.append({
                "features": {
                    "priority": td.get("target_priority", sc.get("priority", td.get("agreement", 0.5))),
                    "geometry": td.get("geometry_margin", 0.5),
                    "duration": min(td.get("duration_margin", 0.5), dur_clip),
                    "imagery":  min(td.get("imagery_support", 0.5), img_clip),
                    "clarity":  td.get("clarity_support", 0.5),
                },
                "util": _REALIZED[act], "action": act,
            })
    w = _run(traces_c)
    mae = _mae(w, traces_c)
    print(f"  {dur_clip:>8.1f}  {img_clip:>8.1f}  {mae:>8.5f}  {w['clarity']:>8.4f}"
          f"  {w['geometry']:>9.4f}  {w['duration']:>9.4f}  {w['imagery']:>8.4f}")


# ── 6. CROSS-DATASET CONVERGENCE ──────────────────────────────────────────────
print("\n" + "="*60)
print("6. CROSS-DATASET CONVERGENCE (encounter vs OVL traces)")
print("="*60)
w_enc = _run(ENC)
w_ovl = _run(OVL)
print(f"\n  {'Feature':<12}  {'Prior':>7}  {'Encounter':>10}  {'OVL':>8}  {'Agree?':>8}")
print("  " + "-"*50)
for k in KEYS:
    d_enc = w_enc[k] - DEFAULT_WEIGHTS[k]
    d_ovl = w_ovl[k] - DEFAULT_WEIGHTS[k]
    agree = "YES" if np.sign(d_enc) == np.sign(d_ovl) else "NO"
    print(f"  {k:<12}  {DEFAULT_WEIGHTS[k]:>7.4f}  {w_enc[k]:>10.4f}  {w_ovl[k]:>8.4f}  {agree:>8}")
print(f"\n  MAE(enc): {_mae(w_enc, ENC):.5f}   MAE(ovl): {_mae(w_ovl, OVL):.5f}")
print(f"  Weight distance (L2): {np.sqrt(sum((w_enc[k]-w_ovl[k])**2 for k in KEYS)):.5f}")


# ── 7. STREAMING SIMULATION ───────────────────────────────────────────────────
print("\n" + "="*60)
print("7. STREAMING SIMULATION (traces arrive one at a time, no shuffling)")
print("   Simulates production: each encounter updates weights immediately")
print("="*60)
rng = np.random.default_rng(42)
order = rng.permutation(len(ENC))
w = dict(DEFAULT_WEIGHTS)
checkpoints = [10, 25, 50, 100, 150, 200, 256]
init_mae = _mae(dict(DEFAULT_WEIGHTS), ENC)
print(f"\n  Init MAE: {init_mae:.5f}")
print(f"\n  {'After N':>8}  {'MAE':>8}  {'Improv':>8}  {'clarity':>8}  {'geometry':>9}")
print("  " + "-"*46)
for n, idx in enumerate(order, 1):
    w, _ = _step(w, ENC[idx]["features"], ENC[idx]["util"], 0.02)
    if n in checkpoints:
        mae = _mae(w, ENC)
        pct = (init_mae - mae) / init_mae * 100
        print(f"  {n:>8}  {mae:>8.5f}  {pct:>7.1f}%  {w['clarity']:>8.4f}  {w['geometry']:>9.4f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
