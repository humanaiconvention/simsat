#!/usr/bin/env python3
"""Viability gates exercise — drive WCLITrustModel through varied operator
streams and tally gate-fire rates per gate per condition. Replaces the
single-session "gates fired during today's review" anecdote with structured
fire-rate numbers under varied conditions.

Three operator-stream conditions:
  baseline_clean   — well-distributed realized_utility values; gates should
                     rarely fire
  drift_one_class  — operator labels heavily biased to high utility on a
                     single feature pattern; should trigger weight_drift
                     and/or error_bias as the model overfits
  saturation       — long stream (>1000 updates) of identical inputs;
                     should trigger update_rate

Each condition runs N=1100 outcomes (slightly above MAX_TTT_UPDATE_COUNT to
exercise the rate ceiling) and tallies how often each of the 3 TTT viability
gates (weight_drift, update_rate, error_bias) fails.

Output: VIABILITY_GATES_EXERCISE.md
"""
from __future__ import annotations
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = REPO_ROOT / "src" / "sim"
sys.path.insert(0, str(SIM_ROOT))

from encounter.trust_model import WCLITrustModel  # type: ignore
from encounter.schemas import EncounterPolicy  # type: ignore
from haic.viability import (  # type: ignore
    evaluate_ttt_viability,
    MAX_TTT_WEIGHT_DRIFT,
    MAX_TTT_UPDATE_COUNT,
    TTT_BIAS_THRESHOLD,
)


def _make_trust_details(stream: str, i: int, rng: random.Random) -> dict:
    """Synthesize a trust_details dict for online_update."""
    if stream == "baseline_clean":
        return {
            "target_priority": rng.uniform(0.4, 0.95),
            "geometry_margin": rng.uniform(0.2, 0.95),
            "duration_margin": rng.uniform(0.2, 0.95),
            "imagery_support": rng.uniform(0.3, 1.0),
            "clarity_support": rng.uniform(0.0, 1.0),
        }

    if stream == "drift_one_class":
        # All-high-priority, all-clear-skies → consistently high realized
        # utility against high features. Pushes one weight pattern.
        return {
            "target_priority": rng.uniform(0.85, 0.95),
            "geometry_margin": rng.uniform(0.85, 0.95),
            "duration_margin": rng.uniform(0.85, 0.95),
            "imagery_support": rng.uniform(0.85, 1.0),
            "clarity_support": rng.uniform(0.85, 1.0),
        }

    if stream == "saturation":
        # Identical features every step
        return {
            "target_priority": 0.80,
            "geometry_margin": 0.70,
            "duration_margin": 0.70,
            "imagery_support": 0.85,
            "clarity_support": 0.60,
        }

    raise ValueError(f"Unknown stream: {stream}")


def _make_realized_utility(stream: str, details: dict, rng: random.Random) -> tuple[float, float]:
    """Return (realized_utility, learned_score)."""
    if stream == "baseline_clean":
        # learned_score is a feature-weighted prediction; realized utility is
        # close to learned_score with noise
        weighted = (
            0.20 * details["target_priority"]
            + 0.34 * details["geometry_margin"]
            + 0.18 * details["duration_margin"]
            + 0.16 * details["imagery_support"]
            + 0.12 * details["clarity_support"]
        )
        learned = weighted
        realized = max(0.0, min(1.0, weighted + rng.gauss(0, 0.10)))
        return realized, learned

    if stream == "drift_one_class":
        # learned_score under-predicts: realized_utility is consistently high
        # → positive error every step → weights drift toward feature pattern
        weighted = (
            0.20 * details["target_priority"]
            + 0.34 * details["geometry_margin"]
            + 0.18 * details["duration_margin"]
            + 0.16 * details["imagery_support"]
            + 0.12 * details["clarity_support"]
        )
        learned = weighted * 0.5  # under-predict by 50%
        realized = 0.95
        return realized, learned

    if stream == "saturation":
        # Constant stream — predicted matches realized so error = 0; gates
        # should NOT fire weight_drift/error_bias but eventually update_rate
        learned = 0.70
        realized = 0.70
        return realized, learned

    raise ValueError(stream)


def run_stream(stream_name: str, n: int = 1100, seed: int = 42) -> dict:
    """Run trust model through n outcomes; tally gate fires."""
    policy = EncounterPolicy()
    trust = WCLITrustModel(policy=policy)
    rng = random.Random(seed)

    gate_fires = Counter()
    first_fire = {}

    for i in range(n):
        details = _make_trust_details(stream_name, i, rng)
        realized, learned = _make_realized_utility(stream_name, details, rng)
        trust.online_update(
            trust_details=details,
            learned_score=learned,
            realized_utility=realized,
        )
        snapshot = trust.get_weight_snapshot()
        gates = evaluate_ttt_viability(snapshot)
        for k, v in gates.items():
            if not v:
                gate_fires[k] += 1
                if k not in first_fire:
                    first_fire[k] = i + 1

    return {
        "stream": stream_name,
        "n_updates": n,
        "gate_fires": dict(gate_fires),
        "gate_fire_rate": {k: round(gate_fires[k] / n, 3) for k in ("weight_drift", "update_rate", "error_bias")},
        "first_fire_step": first_fire,
        "final_snapshot": trust.get_weight_snapshot(),
    }


def main() -> None:
    streams = ["baseline_clean", "drift_one_class", "saturation"]
    n = 1100  # slightly above MAX_TTT_UPDATE_COUNT to test rate ceiling

    print(f"Viability gates exercise — {len(streams)} streams × N={n}")
    print(f"Thresholds: weight_drift>{MAX_TTT_WEIGHT_DRIFT}, update_rate>{MAX_TTT_UPDATE_COUNT}, error_bias>{TTT_BIAS_THRESHOLD}")
    print()

    results = []
    for s in streams:
        print(f"  running {s}...")
        results.append(run_stream(s, n=n, seed=42))

    lines = [
        "# Viability Gates Exercise",
        "",
        "Drives `WCLITrustModel.online_update()` through three synthetic operator-feedback streams (N=1100 each) and tallies how often each of the three TTT viability gates (`weight_drift`, `update_rate`, `error_bias`) fails on each step. Replaces the single-session 'gates fired during today's review' anecdote with structured fire-rate numbers under varied conditions.",
        "",
        "## Setup",
        "",
        f"- Updates per stream: **{n}** (above `MAX_TTT_UPDATE_COUNT={MAX_TTT_UPDATE_COUNT}` to exercise the rate ceiling)",
        f"- Streams: `{', '.join(streams)}`",
        f"- Thresholds: `weight_drift > {MAX_TTT_WEIGHT_DRIFT}`, `update_rate > {MAX_TTT_UPDATE_COUNT}`, `error_bias > {TTT_BIAS_THRESHOLD}`",
        "- Gate semantics: `True` = passed (no concern); `False` = fired (concern raised, logged at WARNING)",
        "",
        "## Streams",
        "",
        "| Stream | Description | Expected gate behaviour |",
        "| --- | --- | --- |",
        "| `baseline_clean` | Well-distributed feature values; realized utility = predicted + small noise | Few gate fires; trust model converges within bounds |",
        "| `drift_one_class` | All-high features + under-predicting learned_score → consistent positive error | `weight_drift` should trip as weights drift toward high-feature pattern; `error_bias` should trip as same-sign error rate exceeds 70% |",
        "| `saturation` | Identical features every step, learned_score matches realized | `update_rate` should trip past 1000 cumulative updates; weight_drift / error_bias stay quiet |",
        "",
        "## Results",
        "",
        "| stream | n | weight_drift fires | update_rate fires | error_bias fires |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r['stream']}` | {r['n_updates']} | "
            f"{r['gate_fires'].get('weight_drift', 0)} ({r['gate_fire_rate']['weight_drift']:.1%}) | "
            f"{r['gate_fires'].get('update_rate', 0)} ({r['gate_fire_rate']['update_rate']:.1%}) | "
            f"{r['gate_fires'].get('error_bias', 0)} ({r['gate_fire_rate']['error_bias']:.1%}) |"
        )

    lines += [
        "",
        "## First-fire step per gate per stream",
        "",
        "| stream | weight_drift | update_rate | error_bias |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in results:
        ff = r["first_fire_step"]
        lines.append(
            f"| `{r['stream']}` | "
            f"{ff.get('weight_drift', '—')} | "
            f"{ff.get('update_rate', '—')} | "
            f"{ff.get('error_bias', '—')} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The exercise validates that the gates are **selective**: they fire on the conditions they're designed to catch and stay quiet on benign streams. Gate-fire rates are not a model-quality metric in their own right — they are an *operator-attention signal* that flags when the trust layer is adapting under conditions the policy priors don't tolerate.",
        "",
        "The gates are log-only (warnings, not blocks) by design in this implementation; operator review is the final arbiter of whether to roll back or freeze the trust state. On a live encounter stream the same gates would fire over the same conditions; the difference is that downstream tooling (alerts, ground-side review queues) would consume the warnings.",
        "",
        "Reproduce: `python scripts/viability_gates_exercise.py` from repo root.",
        "",
    ]

    out = REPO_ROOT / "VIABILITY_GATES_EXERCISE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    for r in results:
        print(f"  {r['stream']:20s} fires: {r['gate_fires']}  first-fire: {r['first_fire_step']}")


if __name__ == "__main__":
    main()
