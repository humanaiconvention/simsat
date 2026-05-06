#!/usr/bin/env python3
"""Real SimSatEnv TTT -- autonomous adaptation on actual stored traces.

Uses trust_details and learned_score from all stored SimSat traces
(exactly what trust_model.online_update() receives in production).
Cycles through the corpus N times with shuffling to simulate continuous
autonomous operation. Updates fire on accept/refine traces only -- defer/skip
produce no realized utility signal in production.

Each cycle:
  1. Shuffle the traces
  2. For each accept/refine trace: call online_update(trust_details, learned_score, realized_utility)
  3. Record weight state after each cycle

Error signal is baseline-centered: global miscalibration is a separate bias
term; weight learning tracks relative quality differences across features.

Output: ttt_real_adaptation.png + printed convergence report.

Usage:
    python scripts/ttt_sim_real.py              # 60 cycles
    python scripts/ttt_sim_real.py --cycles 120
    python scripts/ttt_sim_real.py --no-plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))

TRACES_PATH = REPO_ROOT / "src" / "sim" / "data" / "observation_vla" / "traces.json"

DEFAULT_WEIGHTS = {
    "priority": 0.20,
    "geometry": 0.34,
    "duration": 0.18,
    "imagery":  0.16,
    "clarity":  0.12,
}

# Only actions that produce a realized utility signal (window was opened)
_REALIZED = {"accept": 0.85, "refine": 0.55}


def _load_traces() -> list[dict]:
    traces = json.loads(TRACES_PATH.read_text())["traces"]
    rows = []
    for t in traces:
        da = t.get("decision_after") or {}
        td = da.get("trust_details")
        ls = da.get("learned_score")
        action = (t.get("assessment") or {}).get("recommended_action")
        if td and ls is not None and action in _REALIZED:
            rows.append({
                "trust_details": td,
                "learned_score": float(ls),
                "realized_utility": _REALIZED[action],
                "action": action,
                "pack": t.get("scenario_pack", "unknown"),
            })
    return rows


def _online_step(weights: dict, trust_details: dict, learned_score: float,
                 realized_utility: float, lr: float, reg: float = 0.002) -> tuple[dict, float]:
    """Online gradient step with prior regularization.

    Mirrors trust_model.online_update() with real-data adaptations:
    - duration_margin is always 1.0 in offline replay (saturated feature).
      Clipped to 0.7 so it does not absorb the entire gradient signal.
    - L2 regularization toward DEFAULT_WEIGHTS prevents drift on small corpus.
    - Caller subtracts corpus mean_error before passing realized_utility so the
      error signal is zero-mean (relative quality, not absolute calibration).
    """
    feature_map = {
        "priority": trust_details.get("target_priority", trust_details.get("agreement", 0.5)),
        "geometry": trust_details.get("geometry_margin", 0.5),
        "duration": min(trust_details.get("duration_margin", 0.5), 0.7),
        "imagery":  trust_details.get("imagery_support", 0.5),
        "clarity":  trust_details.get("clarity_support", 0.5),
    }
    error = realized_utility - learned_score
    for k in list(weights):
        grad = error * feature_map.get(k, 0.0)
        reg_pull = reg * (DEFAULT_WEIGHTS[k] - weights[k])
        weights[k] += lr * grad + reg_pull
        weights[k] = max(0.001, weights[k])
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights, error


def run(cycles: int = 60, lr: float = 0.005, seed: int = 42) -> dict:
    traces = _load_traces()
    print(f"Loaded {len(traces)} real traces ({cycles} cycles = {len(traces)*cycles} updates)")

    # Center error signal: weight learning tracks relative quality differences,
    # not the global calibration offset (separate bias term).
    mean_error = float(np.mean([t["realized_utility"] - t["learned_score"] for t in traces]))
    print(f"Corpus mean error: {mean_error:+.4f}  (subtracted as baseline before gradient step)")

    rng = np.random.default_rng(seed)
    weights = dict(DEFAULT_WEIGHTS)
    KEYS = list(DEFAULT_WEIGHTS)

    history: list[dict] = []
    abs_errors: list[float] = []
    update_n = 0

    for cycle in range(cycles):
        order = rng.permutation(len(traces))
        for idx in order:
            row = traces[idx]
            # Pass baseline-centered realized utility
            weights, error = _online_step(
                weights, row["trust_details"], row["learned_score"],
                row["realized_utility"] - mean_error, lr
            )
            abs_errors.append(abs(error))
            update_n += 1

        if cycle % max(1, cycles // 20) == 0 or cycle == cycles - 1:
            history.append({"cycle": cycle + 1, "updates": update_n,
                            **{k: round(weights[k], 5) for k in KEYS}})

    return {
        "final_weights": weights,
        "history": history,
        "abs_errors": abs_errors,
        "n_traces": len(traces),
        "cycles": cycles,
        "lr": lr,
        "total_updates": update_n,
        "mean_error_offset": mean_error,
    }


def print_summary(result: dict) -> None:
    fw = result["final_weights"]
    print(f"\nReal-data TTT -- {result['cycles']} cycles x {result['n_traces']} traces "
          f"= {result['total_updates']} autonomous updates  (lr={result['lr']})")
    print(f"{'Feature':<12}  {'Init':>7}  {'Learned':>8}  {'Drift':>8}")
    print("-" * 42)
    for k in DEFAULT_WEIGHTS:
        drift = fw[k] - DEFAULT_WEIGHTS[k]
        print(f"  {k:<10}  {DEFAULT_WEIGHTS[k]:>7.4f}  {fw[k]:>8.4f}  {drift:>+8.4f}")
    n = len(result["abs_errors"])
    w = max(1, n // 5)
    early = float(np.mean(result["abs_errors"][:w]))
    late  = float(np.mean(result["abs_errors"][-w:]))
    print(f"\nMAE first {w} updates:  {early:.4f}")
    print(f"MAE last  {w} updates:  {late:.4f}")
    pct = (early - late) / early * 100 if early > 0 else 0
    print(f"Relative improvement:  {pct:.1f}%")


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = result["history"]
    updates = [h["updates"] for h in history]
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
        ax.plot(updates, vals, color=COLORS[i], linewidth=2.0, label=k, marker="o",
                markersize=3)
        ax.axhline(DEFAULT_WEIGHTS[k], color=COLORS[i], linewidth=0.7,
                   linestyle="--", alpha=0.4)
    ax.set_title(f"Learned score weight adaptation\n(real SimSat traces, {result['cycles']} cycles)", fontsize=11)
    ax.set_xlabel("Cumulative autonomous updates")
    ax.set_ylabel("Weight (normalised)")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")

    ax = axes[1]
    errs = result["abs_errors"]
    n = len(errs)
    window = max(1, n // 20)
    smooth = np.convolve(errs, np.ones(window) / window, mode="valid")
    ax.plot(range(len(errs)), errs, color="#5dade2", linewidth=0.5, alpha=0.4)
    ax.plot(range(window - 1, n), smooth, color="#e74c3c", linewidth=2.0,
            label=f"{window}-update rolling avg")
    ax.set_title("Prediction error convergence\n(real operator utility signal)", fontsize=11)
    ax.set_xlabel("Autonomous update #")
    ax.set_ylabel("|utility - predicted|")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")

    plt.tight_layout(pad=2.0)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "ttt_real_adaptation.png")
    args = parser.parse_args()

    result = run(cycles=args.cycles, lr=args.lr, seed=args.seed)
    print_summary(result)
    if not args.no_plot:
        plot(result, args.out)


if __name__ == "__main__":
    main()
