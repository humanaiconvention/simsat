#!/usr/bin/env python3
"""Autonomous TTT adaptation simulation — no human in the loop.

Runs N synthetic observation cycles through the trust-layer online_update(),
demonstrating that the learned_score_weights adapt without operator input.

Modes:
  default       Convergence from init → true weights, multi-seed confidence bands
  --shift       Regime-shift: true weights change at cycle N/2 (simulates moving
                from coastal to polar observation zone). Shows the system re-adapts.

Output: ttt_adaptation.png

Usage:
    python scripts/ttt_sim.py --n 5000            # multi-seed convergence plot
    python scripts/ttt_sim.py --n 5000 --shift    # regime-shift demo
    python scripts/ttt_sim.py --no-plot           # text-only
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))

# Default weights from schemas.py — matches production initialisation
DEFAULT_WEIGHTS = {
    "priority": 0.20,
    "geometry": 0.34,
    "duration": 0.18,
    "imagery": 0.16,
    "clarity": 0.12,
}

# Ground-truth feature importances the model must *discover* via TTT.
# Phase 1 (coastal / maritime operations — geometry and imagery dominant)
TRUE_WEIGHTS_P1 = {
    "priority": 0.35,
    "geometry": 0.25,
    "duration": 0.10,
    "imagery":  0.22,
    "clarity":  0.08,
}
# Phase 2 (polar / high-latitude shift — duration and priority become critical,
# imagery less reliable due to polar darkness / weather patterns)
TRUE_WEIGHTS_P2 = {
    "priority": 0.40,
    "geometry": 0.15,
    "duration": 0.28,
    "imagery":  0.10,
    "clarity":  0.07,
}
TRUE_WEIGHTS = TRUE_WEIGHTS_P1  # default (no shift)


def _sample_features(rng: np.random.Generator) -> dict[str, float]:
    return {k: float(rng.beta(2, 2)) for k in DEFAULT_WEIGHTS}


def _true_utility(features: dict[str, float], rng: np.random.Generator,
                  noise: float = 0.08, true_weights: dict | None = None) -> float:
    tw = true_weights or TRUE_WEIGHTS_P1
    score = sum(tw[k] * features[k] for k in tw)
    return float(np.clip(score + rng.normal(0, noise), 0.0, 1.0))


def _learned_score(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[k] * features[k] for k in weights)


def run_single(n: int, lr: float, seed: int, shift: bool = False) -> dict:
    """One seed run. If shift=True, true weights change at cycle n//2."""
    rng = np.random.default_rng(seed)
    weights = dict(DEFAULT_WEIGHTS)
    KEYS = list(DEFAULT_WEIGHTS)
    history: list[dict] = []
    abs_errors: list[float] = []

    for step in range(n):
        true_w = TRUE_WEIGHTS_P2 if (shift and step >= n // 2) else TRUE_WEIGHTS_P1
        features = _sample_features(rng)
        ls = _learned_score(features, weights)
        util = _true_utility(features, rng, true_weights=true_w)
        error = util - ls
        abs_errors.append(abs(error))
        for k in KEYS:
            weights[k] += lr * error * features[k]
            weights[k] = max(0.001, weights[k])
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        if step % max(1, n // 50) == 0:
            history.append({"step": step, **{k: round(weights[k], 5) for k in KEYS}})

    history.append({"step": n, **{k: round(weights[k], 5) for k in KEYS}})
    rolling_mae = [
        float(np.mean(abs_errors[max(0, i - 200):i + 1]))
        for i in range(len(abs_errors))
    ]
    return {"final_weights": weights, "history": history,
            "abs_errors": abs_errors, "rolling_mae": rolling_mae}


def run(n: int = 2000, lr: float = 0.02, seed: int = 42,
        n_seeds: int = 5, shift: bool = False) -> dict:
    seeds = [seed + i for i in range(n_seeds)]
    all_runs = [run_single(n, lr, s, shift=shift) for s in seeds]

    # Aggregate histories onto common step axis
    steps = [h["step"] for h in all_runs[0]["history"]]
    KEYS = list(DEFAULT_WEIGHTS)
    weight_bands: dict[str, dict] = {}
    for k in KEYS:
        mat = np.array([[h[k] for h in r["history"]] for r in all_runs])
        weight_bands[k] = {"mean": mat.mean(0), "lo": mat.min(0), "hi": mat.max(0)}

    # MAE bands
    mae_mat = np.array([r["rolling_mae"] for r in all_runs])
    mae_band = {"mean": mae_mat.mean(0), "lo": mae_mat.min(0), "hi": mae_mat.max(0)}

    ref = all_runs[0]
    return {
        "final_weights": ref["final_weights"],
        "true_weights_p1": TRUE_WEIGHTS_P1,
        "true_weights_p2": TRUE_WEIGHTS_P2,
        "shift": shift,
        "steps": steps,
        "weight_bands": weight_bands,
        "mae_band": mae_band,
        "abs_errors": ref["abs_errors"],
        "n": n,
        "lr": lr,
        "n_seeds": n_seeds,
    }


def print_summary(result: dict) -> None:
    fw = result["final_weights"]
    tw = result["true_weights_p1"]
    label = "regime-shift" if result["shift"] else "convergence"
    print(f"\nTTT adaptation ({label}) — {result['n']} cycles × {result['n_seeds']} seeds  (lr={result['lr']})")
    print(f"{'Feature':<12}  {'Init':>7}  {'Learned':>8}  {'True P1':>8}  {'Error':>8}")
    print("-" * 54)
    for k in DEFAULT_WEIGHTS:
        print(f"  {k:<10}  {DEFAULT_WEIGHTS[k]:>7.4f}  {fw[k]:>8.4f}  {tw[k]:>8.4f}  {fw[k]-tw[k]:>+8.4f}")
    errs = result["abs_errors"]
    w = max(1, len(errs) // 5)
    early, late = float(np.mean(errs[:w])), float(np.mean(errs[-w:]))
    print(f"\nMAE first {w}: {early:.4f}  |  last {w}: {late:.4f}  |  improvement: {(early-late)/early*100:.1f}%")


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = result["steps"]
    wb = result["weight_bands"]
    mb = result["mae_band"]
    KEYS = list(DEFAULT_WEIGHTS)
    COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    shift = result["shift"]
    n = result["n"]

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
        ax.plot(steps, wb[k]["mean"], color=COLORS[i], linewidth=2.0, label=k)
        ax.fill_between(steps, wb[k]["lo"], wb[k]["hi"],
                        color=COLORS[i], alpha=0.15)
        # Phase 1 target (dashed)
        ax.axhline(result["true_weights_p1"][k], color=COLORS[i],
                   linewidth=0.8, linestyle="--", alpha=0.5)
        if shift:
            ax.axhline(result["true_weights_p2"][k], color=COLORS[i],
                       linewidth=0.8, linestyle=":", alpha=0.5)
    if shift:
        ax.axvline(n // 2, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.4)
        ax.text(n // 2 + n * 0.01, ax.get_ylim()[1] * 0.95,
                "regime shift", color="#aaa", fontsize=8)
        title = f"Weight adaptation — regime shift at cycle {n//2}\n(solid=mean, band=min/max over {result['n_seeds']} seeds)"
    else:
        title = f"Weight adaptation — {result['n_seeds']}-seed confidence band\n(dashed = true weights)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Autonomous cycle")
    ax.set_ylabel("Weight (normalised)")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")
    ax.set_xlim(0, n)

    ax = axes[1]
    x_mae = list(range(len(mb["mean"])))
    ax.fill_between(x_mae, mb["lo"], mb["hi"], color="#5dade2", alpha=0.2)
    ax.plot(x_mae, mb["mean"], color="#5dade2", linewidth=1.5, label="mean MAE (rolling 200)")
    if shift:
        ax.axvline(n // 2, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.4)
    ax.set_title("Prediction error convergence\n(no human labels required)", fontsize=11)
    ax.set_xlabel("Autonomous cycle")
    ax.set_ylabel("|utility − predicted|")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")
    ax.set_xlim(0, n)

    plt.tight_layout(pad=2.0)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000, help="Number of autonomous cycles")
    parser.add_argument("--lr", type=float, default=0.02, help="Online learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib output")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "ttt_adaptation.png")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds for confidence band")
    parser.add_argument("--shift", action="store_true",
                        help="Regime-shift demo: true weights change at cycle N/2")
    args = parser.parse_args()

    label = "regime-shift" if args.shift else "convergence"
    print(f"Running {args.n} cycles × {args.n_seeds} seeds ({label}) ...")
    result = run(n=args.n, lr=args.lr, seed=args.seed, n_seeds=args.n_seeds, shift=args.shift)
    print_summary(result)

    if not args.no_plot:
        try:
            plot(result, args.out)
        except ImportError:
            print("matplotlib not available — run with --no-plot to suppress")


if __name__ == "__main__":
    main()
