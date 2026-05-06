#!/usr/bin/env python3
"""SimSatEnv episode reward comparison: default vs TTT-learned weights.

Drives SimSatEnv (offline_replay) using two trust-score policies:
  1. Default weights  (priority=0.20, geometry=0.34, duration=0.18,
                       imagery=0.16, clarity=0.12)
  2. TTT-learned weights (obtained by running ttt_sim_env.run(20 cycles))

Policy: compute weighted trust score from the five encounter features.
  score >= THRESHOLD (0.65) → TRUST_ACCEPT
  score <  THRESHOLD        → TRUST_SKIP

Corpus: 544 records with operator-labelled outcomes (accept/refine → useful=True,
skip/defer → useful=False).  Each episode = one full shuffled pass.

Key signal: clarity (+0.45) and imagery (+0.25) separate accept from skip records;
geometry (delta ≈ 0) does not.  TTT learns to up-weight clarity/imagery and
down-weight geometry — this should reduce false-accepts and improve episode reward.

Usage:
    python scripts/ttt_env_comparison.py             # 20 episodes, produce plot
    python scripts/ttt_env_comparison.py --episodes 40
    python scripts/ttt_env_comparison.py --no-plot
    python scripts/ttt_env_comparison.py --threshold 0.60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sim.muzero.simsat_env import (
    EncounterRecord, SimSatEnv,
    TRUST_ACCEPT, TRUST_SKIP,
    OBS_H, OBS_W,
)

RECORDS_PATH = REPO_ROOT / "src" / "sim" / "data" / "encounter" / "records.json"

DEFAULT_WEIGHTS = {
    "priority": 0.20,
    "geometry": 0.34,
    "duration": 0.18,
    "imagery":  0.16,
    "clarity":  0.12,
}

THRESHOLD = 0.65

# Maps observation_recommended_action → (useful, realized_utility)
_ACTION_MAP = {
    "accept": (True,  None),   # utility = learned_score (continuous)
    "refine": (True,  None),   # lower-confidence accept
    "skip":   (False, None),
    "defer":  (False, None),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_features(r: dict) -> dict:
    """Pull the five trust features from a raw records.json entry."""
    d   = r.get("decision", {})
    td  = d.get("trust_details", {})
    sc  = r.get("features", {}).get("score_components", {})
    return {
        "priority": td.get("target_priority",
                            sc.get("priority", td.get("agreement", 0.5))),
        "geometry": td.get("geometry_margin", 0.5),
        "duration": min(td.get("duration_margin", 0.5), 0.7),
        "imagery":  min(td.get("imagery_support", 0.5), 0.7),
        "clarity":  td.get("clarity_support", 0.5),
    }


def load_labeled_records() -> tuple[list[EncounterRecord], list[dict]]:
    """Return (EncounterRecord list, feature dict list) for the 544 labeled records.

    Only records with operator-assigned observation_recommended_action are included.
    Tile is synthetic zeros — reward is determined by useful/utility_realized metadata.
    """
    raw = json.loads(RECORDS_PATH.read_text())["records"]
    enc_records: list[EncounterRecord] = []
    feat_list:   list[dict]            = []
    synthetic_tile = np.zeros((1, OBS_H, OBS_W), dtype=np.float32)

    for r in raw:
        d   = r.get("decision", {})
        act = d.get("observation_recommended_action")
        ls  = d.get("learned_score")
        if act not in _ACTION_MAP or ls is None:
            continue
        useful, _ = _ACTION_MAP[act]
        # utility_realized is set for genuinely useful records; None for skip/defer
        utility_realized = float(ls) if useful else None

        w = r.get("window", {})
        p = r.get("probe",  {})
        g = w.get("geometry", {})

        enc = EncounterRecord(
            window_id  = d.get("window_id", r.get("window", {}).get("window_id", "?")),
            target_id  = d.get("target_id", "unknown"),
            scenario_pack = "all",
            tile       = synthetic_tile,
            target_priority = str(w.get("target_priority", "normal")),
            sentinel_available   = p.get("sentinel_available"),
            sentinel_cloud_cover = p.get("sentinel_cloud_cover"),
            elevation_degrees    = g.get("elevation_degrees"),
            line_of_sight        = g.get("target_visible"),
            actual_action        = act,
            utility_realized     = utility_realized,
            useful               = useful,
        )
        enc_records.append(enc)
        feat_list.append(_extract_features(r))

    return enc_records, feat_list


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _score(weights: dict, features: dict) -> float:
    return sum(weights[k] * features[k] for k in weights)


def policy(weights: dict, features: dict, threshold: float) -> int:
    return TRUST_ACCEPT if _score(weights, features) >= threshold else TRUST_SKIP


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(
    weights: dict,
    enc_records: list[EncounterRecord],
    feat_list:   list[dict],
    n_episodes:  int,
    threshold:   float,
    seed:        int,
) -> dict:
    """Run n_episodes shuffled passes; return per-episode stats."""
    rng = np.random.default_rng(seed)
    n = len(enc_records)

    episode_rewards: list[float]  = []
    episode_accepts: list[int]    = []
    episode_skips:   list[int]    = []
    true_accepts:    list[int]    = []   # accept when useful=True
    false_accepts:   list[int]    = []  # accept when useful=False

    for ep in range(n_episodes):
        order = rng.permutation(n)
        shuffled_enc  = [enc_records[i] for i in order]
        shuffled_feat = [feat_list[i]   for i in order]

        env = SimSatEnv(backend="offline_replay", records=shuffled_enc, seed=None)
        env.reset()

        ep_reward = 0.0
        ep_accept = ep_skip = ep_ta = ep_fa = 0
        done = False
        step = 0
        while not done:
            act = policy(weights, shuffled_feat[step], threshold)
            _, reward, done, _ = env.step(act)
            ep_reward += reward
            if act == TRUST_ACCEPT:
                ep_accept += 1
                if shuffled_enc[step].useful:
                    ep_ta += 1
                elif shuffled_enc[step].useful is False:
                    ep_fa += 1
            else:
                ep_skip += 1
            step += 1

        episode_rewards.append(ep_reward)
        episode_accepts.append(ep_accept)
        episode_skips.append(ep_skip)
        true_accepts.append(ep_ta)
        false_accepts.append(ep_fa)

    return {
        "episode_rewards":  episode_rewards,
        "mean_reward":      float(np.mean(episode_rewards)),
        "std_reward":       float(np.std(episode_rewards)),
        "mean_accepts":     float(np.mean(episode_accepts)),
        "mean_skips":       float(np.mean(episode_skips)),
        "mean_true_accepts":  float(np.mean(true_accepts)),
        "mean_false_accepts": float(np.mean(false_accepts)),
        "n_records":        n,
        "n_episodes":       n_episodes,
        "threshold":        threshold,
        "weights":          weights,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(default_res: dict, learned_res: dict) -> None:
    d = default_res
    l = learned_res
    pct = (l["mean_reward"] - d["mean_reward"]) / abs(d["mean_reward"]) * 100 if d["mean_reward"] != 0 else float("nan")

    print(f"\n{'='*65}")
    print(f"SimSatEnv episode comparison — {d['n_episodes']} episodes × "
          f"{d['n_records']} records  (threshold={d['threshold']})")
    print(f"{'='*65}")
    print(f"\n{'Metric':<30}  {'Default':>12}  {'Learned':>12}  {'Delta':>10}")
    print("-" * 65)

    rows = [
        ("Mean episode reward",    d["mean_reward"],      l["mean_reward"],      True),
        ("  std",                  d["std_reward"],       l["std_reward"],       False),
        ("Mean accepts/ep",        d["mean_accepts"],     l["mean_accepts"],     False),
        ("  true accepts",         d["mean_true_accepts"],l["mean_true_accepts"],False),
        ("  false accepts",        d["mean_false_accepts"],l["mean_false_accepts"],False),
        ("Mean skips/ep",          d["mean_skips"],       l["mean_skips"],       False),
    ]
    for label, dv, lv, highlight in rows:
        delta = lv - dv
        flag = " <--" if highlight and abs(delta) > 0.1 else ""
        print(f"  {label:<28}  {dv:>12.3f}  {lv:>12.3f}  {delta:>+10.3f}{flag}")

    print(f"\n  Reward improvement: {pct:+.1f}%")

    print(f"\n{'Feature':<12}  {'Default':>8}  {'Learned':>8}  {'Drift':>8}")
    print("-" * 42)
    for k in DEFAULT_WEIGHTS:
        lw = l["weights"][k]
        drift = lw - DEFAULT_WEIGHTS[k]
        flag = " <--" if abs(drift) > 0.01 else ""
        print(f"  {k:<10}  {DEFAULT_WEIGHTS[k]:>8.4f}  {lw:>8.4f}  {drift:>+8.4f}{flag}")
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(default_res: dict, learned_res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d_rewards = default_res["episode_rewards"]
    l_rewards = learned_res["episode_rewards"]
    eps = list(range(1, len(d_rewards) + 1))

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

    # Left: per-episode reward traces
    ax = axes[0]
    ax.plot(eps, d_rewards, color="#3498db", linewidth=2.0, marker="o", markersize=4,
            label=f"default  (μ={default_res['mean_reward']:.2f})")
    ax.plot(eps, l_rewards, color="#2ecc71", linewidth=2.0, marker="o", markersize=4,
            label=f"learned  (μ={learned_res['mean_reward']:.2f})")
    ax.axhline(default_res["mean_reward"], color="#3498db", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(learned_res["mean_reward"], color="#2ecc71", linewidth=0.7, linestyle="--", alpha=0.5)
    pct = (learned_res["mean_reward"] - default_res["mean_reward"]) / abs(default_res["mean_reward"]) * 100
    ax.set_title(f"Episode reward: default vs TTT-learned\n"
                 f"({default_res['n_records']} records/ep, threshold={THRESHOLD}, {pct:+.1f}%)",
                 fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")

    # Right: decision breakdown bar chart
    ax = axes[1]
    labels = ["true\naccepts", "false\naccepts", "skips"]
    d_vals = [default_res["mean_true_accepts"], default_res["mean_false_accepts"],
              default_res["mean_skips"]]
    l_vals = [learned_res["mean_true_accepts"], learned_res["mean_false_accepts"],
              learned_res["mean_skips"]]
    x = np.arange(len(labels))
    width = 0.35
    bars_d = ax.bar(x - width/2, d_vals, width, label="default", color="#3498db", alpha=0.85)
    bars_l = ax.bar(x + width/2, l_vals, width, label="learned", color="#2ecc71", alpha=0.85)
    ax.set_title("Decision breakdown (mean per episode)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#aaa")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")
    for bar in list(bars_d) + list(bars_l):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.0f}",
                ha="center", va="bottom", color="#aaa", fontsize=8)

    plt.tight_layout(pad=2.0)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes",  type=int,   default=20)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--no-plot",   action="store_true")
    parser.add_argument("--out",       type=Path,
                        default=REPO_ROOT / "ttt_env_comparison.png")
    args = parser.parse_args()

    # Get exact TTT-learned weights by running ttt_sim_env
    import ttt_sim_env
    print("Running TTT (20 cycles) to get learned weights...")
    ttt_result = ttt_sim_env.run(cycles=20, lr=0.02)
    LEARNED_WEIGHTS = ttt_result["final_weights"]
    ttt_sim_env.print_summary(ttt_result)

    print(f"\nLoading labeled encounter records from:\n  {RECORDS_PATH}")
    enc_records, feat_list = load_labeled_records()
    n_useful   = sum(1 for r in enc_records if r.useful is True)
    n_not_useful = sum(1 for r in enc_records if r.useful is False)
    print(f"Loaded {len(enc_records)} labeled records  "
          f"({n_useful} useful, {n_not_useful} not-useful)")

    print(f"\nRunning {args.episodes} episodes with default weights...")
    default_res = run_episodes(DEFAULT_WEIGHTS, enc_records, feat_list,
                               args.episodes, args.threshold, args.seed)

    print(f"Running {args.episodes} episodes with learned weights...")
    learned_res = run_episodes(LEARNED_WEIGHTS, enc_records, feat_list,
                               args.episodes, args.threshold, args.seed)

    print_summary(default_res, learned_res)

    if not args.no_plot:
        plot(default_res, learned_res, args.out)


if __name__ == "__main__":
    main()
