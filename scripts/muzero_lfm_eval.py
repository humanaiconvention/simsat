#!/usr/bin/env python3
"""LFM Track eval — SimSatEnv + SigLIP tile encoder (Liquid Track).

Demonstrates the complete Liquid Track pipeline end-to-end:

  Sentinel PNG → SigLIP encoder (768-dim) → SimSatEnv episode
    → trust action → reward → planner stats

Uses stored VLA assessment recommended_action as the policy so results
are directly comparable to the General AI Track VLA eval numbers.

Run:
    python scripts/muzero_lfm_eval.py
    python scripts/muzero_lfm_eval.py --scenario maritime_chokepoints
    python scripts/muzero_lfm_eval.py --markdown

Env vars (optional):
    TILE_ENCODER_MODEL   HF model id (default: google/siglip-base-patch16-224)
    TILE_ENCODER_DEVICE  cuda | cpu (default: cuda if available)
    RUN_HF_TILE_ENCODER_TESTS=1   required to download SigLIP (~370 MB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ── Repo path wiring ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "sim"))

from muzero.simsat_env import (
    EncounterRecord,
    SimSatEnv,
    TRUST_ACCEPT,
    TRUST_DEFER,
    TRUST_SKIP,
    TRUST_REFINE,
    TRUST_ACTION_NAMES,
)

# ── Constants ────────────────────────────────────────────────────────────────
TRACES_JSON = REPO_ROOT / "src" / "sim" / "data" / "observation_vla" / "traces.json"
ASSETS_DIR  = REPO_ROOT / "review_queue_assets"

ACTION_MAP = {
    "accept": TRUST_ACCEPT,
    "defer":  TRUST_DEFER,
    "skip":   TRUST_SKIP,
    "refine": TRUST_REFINE,
}

SCENARIO_DISPLAY = {
    "maritime_chokepoints":     "Maritime Chokepoints",
    "disaster_response_weather": "Disaster / Weather",
    "urban_coastal_ambiguity":  "Urban Coastal Ambiguity",
    "all":                      "All Packs",
}


# ── Tile image loading ───────────────────────────────────────────────────────

def _find_tile_path(trace_id: str, scenario_pack: str) -> Path | None:
    """Resolve the PNG for a given trace (format: {scenario_pack}_{trace_id}.png)."""
    candidate = ASSETS_DIR / f"{scenario_pack}_{trace_id}.png"
    if candidate.exists():
        return candidate
    # Fallback: search by trace_id suffix only (handles pack mismatch)
    for p in ASSETS_DIR.glob(f"*{trace_id}*.png"):
        return p
    return None


def _load_tile(path: Path) -> np.ndarray:
    """Load a PNG as a (64, 64) float32 numpy array normalized to [0, 1]."""
    img = Image.open(path).convert("L")   # greyscale
    img = img.resize((64, 64), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ── Record builder ───────────────────────────────────────────────────────────

def build_records(scenario_pack: str = "all") -> tuple[list[EncounterRecord], list[str]]:
    """
    Return (records, policy_actions) from the stored 86-trace corpus.

    policy_actions[i] is the recommended_action from the stored VLA assessment
    for records[i] — used as the planning policy so eval results are comparable
    to the General AI Track VLA eval.
    """
    raw = json.loads(TRACES_JSON.read_text())["traces"]
    records: list[EncounterRecord] = []
    policy_actions: list[str] = []

    for t in raw:
        pack = t.get("scenario_pack", "")
        if scenario_pack not in ("all", pack):
            continue

        tile_path = _find_tile_path(t["trace_id"], pack)
        if tile_path is None:
            continue   # skip traces without an image

        tile = _load_tile(tile_path)

        probe = t.get("probe", {})
        sample = t.get("sample", {})
        ass = t.get("assessment", {})
        evidence = ass.get("evidence", {})

        rec = EncounterRecord(
            window_id=t.get("window_id", t["trace_id"]),
            target_id=t.get("target_id", "unknown"),
            scenario_pack=pack,
            tile=tile,
            target_priority=sample.get("target_priority", "normal"),
            sentinel_available=probe.get("sentinel_available"),
            sentinel_cloud_cover=probe.get("sentinel_cloud_cover"),
            elevation_degrees=probe.get("elevation_degrees"),
            line_of_sight=probe.get("line_of_sight"),
        )
        records.append(rec)
        policy_actions.append(ass.get("recommended_action") or "defer")

    return records, policy_actions


# ── Encoder setup ────────────────────────────────────────────────────────────

def _build_encoder():
    """Load HFVisionTowerEncoder (SigLIP-base or env-var override)."""
    model_id = os.environ.get("TILE_ENCODER_MODEL", "google/siglip-base-patch16-224")
    device = os.environ.get("TILE_ENCODER_DEVICE") or ("cuda" if _cuda_available() else "cpu")
    from muzero.tile_encoder import build_encoder
    print(f"Loading tile encoder: {model_id} on {device} ...")
    t0 = time.perf_counter()
    enc = build_encoder("lfm2vl", model_id=model_id, device=device)
    elapsed = time.perf_counter() - t0
    print(f"Encoder ready: embed_dim={enc.embed_dim}  load_time={elapsed:.1f}s")
    return enc


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episodes(
    records: list[EncounterRecord],
    policy_actions: list[str],
    encoder,
    scenario_pack: str,
) -> dict[str, Any]:
    """
    Run one episode per record using the stored VLA policy action.
    Returns a stats dict comparable to encounter_eval's trust_summary.
    """
    env = SimSatEnv(
        backend="offline_replay",
        records=records,
        scenario_pack="all",   # already filtered
        tile_encoder=encoder,
    )

    action_counts: dict[str, int] = {k: 0 for k in TRUST_ACTION_NAMES.values()}
    rewards: list[float] = []
    embed_times_ms: list[float] = []
    useful_accepts: int = 0
    total_accepts: int = 0

    for i, rec in enumerate(records):
        # One-step episode: reset → single action → done
        env._records = [rec]   # swap to single record
        env.reset()

        action_name = policy_actions[i]
        action_int = ACTION_MAP.get(action_name, TRUST_DEFER)

        # Time the encoder call (it's invoked inside env.reset via _tile_to_obs)
        t0 = time.perf_counter()
        env.reset()   # re-run to get timing (encoder called in reset)
        embed_times_ms.append((time.perf_counter() - t0) * 1000)

        _, reward, _, info = env.step(action_int)

        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        rewards.append(reward)
        if action_name == "accept":
            total_accepts += 1
            if reward > 0:
                useful_accepts += 1

    n = len(records)
    mat_yield = useful_accepts / total_accepts if total_accepts else None

    return {
        "scenario_pack": scenario_pack,
        "window_count": n,
        "action_counts": action_counts,
        "materialization_yield": mat_yield,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std":  float(np.std(rewards))  if rewards else 0.0,
        "encoder_ms_mean": float(np.mean(embed_times_ms)) if embed_times_ms else 0.0,
        "encoder_ms_p95":  float(np.percentile(embed_times_ms, 95)) if embed_times_ms else 0.0,
        "embed_dim": encoder.embed_dim,
        "encoder_model": encoder.model_id,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "-"


def print_results(stats: dict, markdown: bool = False) -> None:
    pack_label = SCENARIO_DISPLAY.get(stats["scenario_pack"], stats["scenario_pack"])
    n = stats["window_count"]
    ac = stats["action_counts"]
    enc_ms = stats["encoder_ms_mean"]
    enc_p95 = stats["encoder_ms_p95"]
    mat = stats["materialization_yield"]
    mat_str = f"{mat:.2f}" if mat is not None else "-"

    if markdown:
        print(f"\n## LFM Track — {pack_label}\n")
        print(f"| Metric | Value |")
        print(f"|---|---|")
        print(f"| Encoder | `{stats['encoder_model']}` |")
        print(f"| Embed dim | {stats['embed_dim']} |")
        print(f"| Episodes | {n} |")
        print(f"| Encoder latency (mean / p95) | {enc_ms:.1f} ms / {enc_p95:.1f} ms |")
        print(f"| Action: accept | {ac.get('accept',0)} ({_pct(ac.get('accept',0), n)}) |")
        print(f"| Action: refine | {ac.get('refine',0)} ({_pct(ac.get('refine',0), n)}) |")
        print(f"| Action: defer  | {ac.get('defer',0)}  ({_pct(ac.get('defer',0),  n)}) |")
        print(f"| Action: skip   | {ac.get('skip',0)}   ({_pct(ac.get('skip',0),   n)}) |")
        print(f"| Materialization yield | {mat_str} |")
        print(f"| Mean episode reward | {stats['reward_mean']:.4f} |")
    else:
        print(f"\nLFM Track Eval — {pack_label}")
        print(f"  Encoder:      {stats['encoder_model']}  (embed_dim={stats['embed_dim']})")
        print(f"  Latency:      {enc_ms:.1f} ms/tile (p95 {enc_p95:.1f} ms)")
        print(f"  Episodes:     {n}")
        print(f"  Actions:      accept={ac.get('accept',0)}  refine={ac.get('refine',0)}"
              f"  defer={ac.get('defer',0)}  skip={ac.get('skip',0)}")
        print(f"  Mat. yield:   {mat_str}")
        print(f"  Mean reward:  {stats['reward_mean']:.4f} ± {stats['reward_std']:.4f}")


def write_markdown(stats_list: list[dict]) -> None:
    out = Path("MUZERO_LFM_EVAL.md")
    lines = [
        "# LFM Track Eval — MuZero + SigLIP Tile Encoder\n",
        "Pipeline: `Sentinel PNG → SigLIP-base (768-dim) → SimSatEnv → trust action → reward`  ",
        "Policy: stored VLA `recommended_action` from the 86-trace corpus.  ",
        f"LFM2.5-VL encoder slot: one-line `model_id` swap in `build_encoder(\"lfm2vl\", model_id=...)`.  ",
        "",
    ]
    for s in stats_list:
        pack_label = SCENARIO_DISPLAY.get(s["scenario_pack"], s["scenario_pack"])
        n = s["window_count"]
        ac = s["action_counts"]
        mat = s["materialization_yield"]
        mat_str = f"{mat:.2f}" if mat is not None else "-"
        lines += [
            f"## {pack_label}",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Encoder | `{s['encoder_model']}` |",
            f"| Embed dim | {s['embed_dim']} |",
            f"| Episodes (matched tiles) | {n} |",
            f"| Encoder latency mean / p95 | {s['encoder_ms_mean']:.1f} ms / {s['encoder_ms_p95']:.1f} ms |",
            f"| accept | {ac.get('accept',0)} ({_pct(ac.get('accept',0), n)}) |",
            f"| refine | {ac.get('refine',0)} ({_pct(ac.get('refine',0), n)}) |",
            f"| defer  | {ac.get('defer',0)} ({_pct(ac.get('defer',0), n)}) |",
            f"| skip   | {ac.get('skip',0)} ({_pct(ac.get('skip',0), n)}) |",
            f"| Materialization yield | {mat_str} |",
            f"| Mean episode reward | {s['reward_mean']:.4f} ± {s['reward_std']:.4f} |",
            "",
        ]
    out.write_text("\n".join(lines))
    print(f"\nResults written to {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LFM Track eval — MuZero + tile encoder")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "maritime_chokepoints",
                                 "disaster_response_weather", "urban_coastal_ambiguity"])
    parser.add_argument("--markdown", action="store_true",
                        help="Print markdown table output")
    parser.add_argument("--write-md", action="store_true",
                        help="Write MUZERO_LFM_EVAL.md to repo root")
    args = parser.parse_args()

    if os.environ.get("RUN_HF_TILE_ENCODER_TESTS") != "1":
        print("Note: SigLIP weights (~370 MB) will be downloaded on first run.")
        print("Set RUN_HF_TILE_ENCODER_TESTS=1 to suppress this message.\n")

    encoder = _build_encoder()

    packs = (
        ["maritime_chokepoints", "disaster_response_weather", "urban_coastal_ambiguity"]
        if args.scenario == "all"
        else [args.scenario]
    )

    all_stats: list[dict] = []
    for pack in packs:
        records, policy_actions = build_records(pack)
        if not records:
            print(f"  No matched tiles for {pack}, skipping.")
            continue
        stats = run_episodes(records, policy_actions, encoder, pack)
        print_results(stats, markdown=args.markdown)
        all_stats.append(stats)

    if args.scenario == "all" and len(all_stats) > 1:
        # Aggregate
        total_n = sum(s["window_count"] for s in all_stats)
        agg_ac: dict[str, int] = {}
        for s in all_stats:
            for k, v in s["action_counts"].items():
                agg_ac[k] = agg_ac.get(k, 0) + v
        useful = sum(
            s["action_counts"].get("accept", 0) * (s["materialization_yield"] or 0)
            for s in all_stats
        )
        total_accepts = sum(s["action_counts"].get("accept", 0) for s in all_stats)
        agg_yield = useful / total_accepts if total_accepts else None
        agg = {
            "scenario_pack": "all",
            "window_count": total_n,
            "action_counts": agg_ac,
            "materialization_yield": agg_yield,
            "reward_mean": float(np.mean([s["reward_mean"] for s in all_stats])),
            "reward_std":  float(np.mean([s["reward_std"]  for s in all_stats])),
            "encoder_ms_mean": float(np.mean([s["encoder_ms_mean"] for s in all_stats])),
            "encoder_ms_p95":  float(np.mean([s["encoder_ms_p95"]  for s in all_stats])),
            "embed_dim": all_stats[0]["embed_dim"],
            "encoder_model": all_stats[0]["encoder_model"],
        }
        print_results(agg, markdown=args.markdown)
        all_stats.append(agg)

    if args.write_md:
        write_markdown(all_stats)


if __name__ == "__main__":
    main()
