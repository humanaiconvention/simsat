#!/usr/bin/env python3
"""TTT on full encounter records corpus -- live prediction recomputation.

Drives TTT online_update() on 256 encounter records with confirmed VLA
utility signal (156 accept + 100 refine from 5907 total records).

Key design: predictions are recomputed with CURRENT weights each cycle,
so MAE tracks actual improvement -- not residuals against stale stored scores.

Results on real SimSat data:
  - MAE drops 19.4% in cycle 1 (0.140 -> 0.113)
  - Converges to 22.4% improvement by cycle 4
  - Clarity: +0.12 (0.12->0.24), Geometry: -0.10 (0.34->0.24)
  - Consistent with ttt_sim_real.py direction across independent datasets

Usage:
    python scripts/ttt_sim_env.py              # 20 cycles
    python scripts/ttt_sim_env.py --cycles 40 --lr 0.02
    python scripts/ttt_sim_env.py --no-plot
    python scripts/ttt_sim_env.py --shift      # coastal->polar regime shift demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))

RECORDS_PATH = REPO_ROOT / "src" / "sim" / "data" / "encounter" / "records.json"

DEFAULT_WEIGHTS = {
    "priority": 0.20,
    "geometry": 0.34,
    "duration": 0.18,
    "imagery":  0.16,
    "clarity":  0.12,
}

_REALIZED = {"accept": 0.85, "refine": 0.55}


def _load_records() -> list[dict]:
    all_records = json.loads(RECORDS_PATH.read_text())["records"]
    rows = []
    for r in all_records:
        d = r.get("decision", {})
        td = d.get("trust_details", {})
        ls = d.get("learned_score")
        act = d.get("observation_recommended_action")
        if td and ls is not None and act in _REALIZED:
            sc = r.get("features", {}).get("score_components", {})
            rows.append({
                "features": {
                    "priority": td.get("target_priority",
                                       sc.get("priority", td.get("agreement", 0.5))),
                    "geometry": td.get("geometry_margin", 0.5),
                    # duration and imagery saturated at 1.0 in high-quality corpus
                    "duration": min(td.get("duration_margin", 0.5), 0.7),
                    "imagery":  min(td.get("imagery_support", 0.5), 0.7),
                    "clarity":  td.get("clarity_support", 0.5),
                },
                "util": _REALIZED[act],
                "action": act,
            })
    return rows


def _predict(weights: dict, features: dict) -> float:
    return sum(weights[k] * features[k] for k in weights)


def _online_step(weights: dict, features: dict, realized_utility: float,
                 lr: float, reg: float = 0.002) -> tuple[dict, float]:
    """Online gradient step using live recomputed prediction."""
    ls_live = _predict(weights, features)
    error = realized_utility - ls_live
    for k in list(weights):
        weights[k] += lr * error * features[k] + reg * (DEFAULT_WEIGHTS[k] - weights[k])
        weights[k] = max(0.001, weights[k])
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights, error


def run(cycles: int = 20, lr: float = 0.02, seed: int = 42) -> dict:
    traces = _load_records()
    n = len(traces)
    print(f"Loaded {n} encounter records  ({cycles} cycles = {n*cycles} updates)")

    rng = np.random.default_rng(seed)
    weights = dict(DEFAULT_WEIGHTS)
    KEYS = list(DEFAULT_WEIGHTS)

    # Measure MAE before any updates (baseline)
    init_mae = float(np.mean([abs(t["util"] - _predict(weights, t["features"])) for t in traces]))

    history: list[dict] = []
    history.append({"cycle": 0, "updates": 0,
                    "mae": round(init_mae, 6),
                    **{k: round(weights[k], 5) for k in KEYS}})

    update_n = 0
    for cycle in range(cycles):
        order = rng.permutation(n)
        cycle_errs = []
        for idx in order:
            t = traces[idx]
            weights, error = _online_step(weights, t["features"], t["util"], lr)
            cycle_errs.append(abs(error))
            update_n += 1

        cycle_mae = float(np.mean([abs(t["util"] - _predict(weights, t["features"])) for t in traces]))
        history.append({"cycle": cycle + 1, "updates": update_n,
                        "mae": round(cycle_mae, 6),
                        **{k: round(weights[k], 5) for k in KEYS}})

    return {
        "final_weights": weights,
        "history": history,
        "init_mae": init_mae,
        "final_mae": history[-1]["mae"],
        "n_traces": n,
        "cycles": cycles,
        "lr": lr,
        "total_updates": update_n,
    }


def print_summary(result: dict) -> None:
    fw = result["final_weights"]
    pct = (result["init_mae"] - result["final_mae"]) / result["init_mae"] * 100
    print(f"\nEncounter-corpus TTT -- {result['cycles']} cycles x {result['n_traces']} records "
          f"= {result['total_updates']} updates  (lr={result['lr']})")
    print(f"MAE:  {result['init_mae']:.5f} -> {result['final_mae']:.5f}  "
          f"({pct:.1f}% improvement)")
    print(f"\n{'Feature':<12}  {'Init':>7}  {'Learned':>8}  {'Drift':>8}")
    print("-" * 42)
    for k in DEFAULT_WEIGHTS:
        drift = fw[k] - DEFAULT_WEIGHTS[k]
        flag = " <--" if abs(drift) > 0.01 else ""
        print(f"  {k:<10}  {DEFAULT_WEIGHTS[k]:>7.4f}  {fw[k]:>8.4f}  {drift:>+8.4f}{flag}")
    print("\nPer-cycle MAE:")
    for h in result["history"]:
        bar = "#" * int(h["mae"] * 100)
        print(f"  cycle {h['cycle']:3d}: {h['mae']:.5f}  {bar}")


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = result["history"]
    cycles_x = [h["cycle"] for h in history]
    maes = [h["mae"] for h in history]
    KEYS = list(DEFAULT_WEIGHTS)
    COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="#888")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#ccc")

    ax = axes[0]
    for i, k in enumerate(KEYS):
        vals = [h[k] for h in history]
        ax.plot(cycles_x, vals, color=COLORS[i], linewidth=2.0, label=k, marker="o", markersize=4)
        ax.axhline(DEFAULT_WEIGHTS[k], color=COLORS[i], linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_title(f"Encounter-corpus weight adaptation\n({result['n_traces']} records, "
                 f"lr={result['lr']})", fontsize=11)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Weight (normalised)")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")

    ax = axes[1]
    ax.plot(cycles_x, maes, color="#e74c3c", linewidth=2.5, marker="o", markersize=5)
    ax.axhline(result["init_mae"], color="#888", linewidth=0.8, linestyle="--", alpha=0.5,
               label=f"init MAE {result['init_mae']:.4f}")
    pct = (result["init_mae"] - result["final_mae"]) / result["init_mae"] * 100
    ax.set_title(f"MAE convergence -- {pct:.1f}% improvement\n"
                 f"(live recomputed predictions, real SimSat data)", fontsize=11)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("MAE (|utility - predicted|)")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")

    plt.tight_layout(pad=2.0)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out}")


def run_shift(cycles_pre: int = 15, cycles_shift: int = 15,
              lr: float = 0.02, seed: int = 42) -> dict:
    """Two-phase regime shift: coastal convergence -> polar transition.

    Phase 1 (cycles_pre): standard corpus -- clarity-heavy coastal scenes.
    Phase 2 (cycles_shift): polar-like synthetic perturbation.
      - clarity_support *= 0.35  (fog/cloud-heavy polar passes)
      - geometry_margin *= 1.35  (polar geometry more favorable)
    Demonstrates autonomous re-adaptation without weight reset.
    """
    traces = _load_records()
    n = len(traces)
    print(f"\nShift demo: {cycles_pre} coastal + {cycles_shift} polar cycles "
          f"on {n} records  (lr={lr})")

    rng = np.random.default_rng(seed)
    weights = dict(DEFAULT_WEIGHTS)
    KEYS = list(DEFAULT_WEIGHTS)

    def _mae(w: dict) -> float:
        return float(np.mean([abs(t["util"] - _predict(w, t["features"])) for t in traces]))

    history: list[dict] = []
    history.append({"phase": "init", "cycle": 0,
                    **{k: round(weights[k], 5) for k in KEYS},
                    "mae": round(_mae(weights), 6)})

    # Phase 1: coastal
    for cycle in range(cycles_pre):
        for idx in rng.permutation(n):
            weights, _ = _online_step(weights, traces[idx]["features"], traces[idx]["util"], lr)
        history.append({"phase": "coastal", "cycle": cycle + 1,
                        **{k: round(weights[k], 5) for k in KEYS},
                        "mae": round(_mae(weights), 6)})

    coastal_final = dict(weights)

    # Phase 2: polar perturbation
    polar_traces = [{
        "features": {
            **t["features"],
            "clarity":  t["features"]["clarity"]  * 0.35,
            "geometry": min(t["features"]["geometry"] * 1.35, 0.99),
        },
        "util": t["util"],
    } for t in traces]

    for cycle in range(cycles_shift):
        for idx in rng.permutation(n):
            weights, _ = _online_step(weights, polar_traces[idx]["features"],
                                      polar_traces[idx]["util"], lr)
        history.append({"phase": "polar", "cycle": cycles_pre + cycle + 1,
                        **{k: round(weights[k], 5) for k in KEYS},
                        "mae": round(_mae(weights), 6)})

    print(f"\n{'Feature':<12}  {'Init':>7}  {'Coastal':>8}  {'Polar':>8}  {'Total drift':>12}")
    print("-" * 52)
    for k in KEYS:
        print(f"  {k:<10}  {DEFAULT_WEIGHTS[k]:>7.4f}  {coastal_final[k]:>8.4f}"
              f"  {weights[k]:>8.4f}  {weights[k]-DEFAULT_WEIGHTS[k]:>+12.4f}")
    return {"history": history, "coastal_weights": coastal_final,
            "polar_weights": weights, "n_traces": n,
            "cycles_pre": cycles_pre, "cycles_shift": cycles_shift, "lr": lr}


def plot_shift(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = result["history"]
    KEYS = list(DEFAULT_WEIGHTS)
    COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    cycles_x = [h["cycle"] for h in history]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111")
    ax.tick_params(colors="#888")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#ccc")

    for i, k in enumerate(KEYS):
        vals = [h[k] for h in history]
        ax.plot(cycles_x, vals, color=COLORS[i], linewidth=2.0, label=k, marker="o", markersize=4)

    boundary = result["cycles_pre"]
    ymin, ymax = ax.get_ylim()
    ax.axvline(boundary, color="#555", linewidth=1.5, linestyle="--")
    ax.text(boundary + 0.3, ymax * 0.97, "polar -->",
            color="#888", fontsize=9)
    ax.set_title(f"TTT regime shift: coastal->polar\n"
                 f"({result['n_traces']} records, lr={result['lr']})", fontsize=11)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Weight (normalised)")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")

    plt.tight_layout(pad=2.0)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "ttt_env_adaptation.png")
    parser.add_argument("--shift", action="store_true",
                        help="Run coastal->polar regime shift demo instead of standard adaptation")
    parser.add_argument("--shift-out", type=Path,
                        default=REPO_ROOT / "ttt_regime_shift.png")
    args = parser.parse_args()

    if args.shift:
        result = run_shift(lr=args.lr, seed=args.seed)
        if not args.no_plot:
            plot_shift(result, args.shift_out)
        return

    result = run(cycles=args.cycles, lr=args.lr, seed=args.seed)
    print_summary(result)
    if not args.no_plot:
        plot(result, args.out)


if __name__ == "__main__":
    main()
