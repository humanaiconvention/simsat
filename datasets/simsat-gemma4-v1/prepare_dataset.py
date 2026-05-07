#!/usr/bin/env python3
"""
Prepare SimSat Gemma-4 v1 training dataset for Kaggle upload.

Reads from <repo_root>/exports/gemma4_v3/ and writes into this directory:
  simsat_train.jsonl       — merged train split in ChatML messages format
  simsat_eval_reviewed.jsonl  — 3 reviewed holdout cases (raw fields + messages)
  simsat_eval_shortlist.jsonl — 10 per-target shortlist cases (raw fields + messages)

Training weight → repetition count (rounded, capped at 10):
  format_train:        weight=1.5 → 2 repeats
  multimodal_weak:     weight=2.0 → 2 repeats (text-only, no image column)
  multimodal_reviewed: weight=6-8 → 6-8 repeats

Set REFINE_BOOST=<float> env var (default 1.0) to upweight `refine` action rows.
Example: REFINE_BOOST=2.0 doubles the repeat count for every refine case.
Use this to address accept-bias in v9 (model over-selects accept on borderline refine).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parents[2] / "exports" / "gemma4_v4"
OUT_DIR = Path(__file__).parent

REFINE_BOOST = float(os.environ.get("REFINE_BOOST", "1.0"))

SYSTEM_PROMPT = (
    "You are a satellite encounter-assessment AI. "
    "Given metadata about a planned Earth observation, return a compact JSON assessment. "
    "Fields: usable_observation (bool), scene_match_score (float 0-1), "
    "salience_score (float 0-1), change_or_event_score (float 0-1), "
    "occlusion_or_cloud_risk (float 0-1), confidence (float 0-1), "
    "recommended_action (accept|defer|refine|skip), rationale_tags (list[str]). "
    "Return only valid JSON, no markdown, no explanation."
)


def to_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["prompt_text"]},
        {"role": "assistant", "content": json.dumps(record["target_json"])},
    ]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(rows)} rows → {path.name}")


def main() -> None:
    print("SimSat Gemma-4 v1 dataset preparation")
    print(f"  Source: {EXPORT_DIR}")
    print(f"  Output: {OUT_DIR}")
    print()

    # ── Train split ──────────────────────────────────────────────────────────
    train_rows: list[dict] = []

    splits = [
        ("simsat_format_train.jsonl", "format_train"),
        ("simsat_multimodal_weak.jsonl", "multimodal_weak"),
        ("simsat_multimodal_reviewed.jsonl", "multimodal_reviewed"),
    ]

    for filename, split_name in splits:
        records = load_jsonl(EXPORT_DIR / filename)
        for rec in records:
            weight = rec.get("training_weight", 1.0)
            action = rec.get("target_json", {}).get("recommended_action", "")
            if action == "refine" and REFINE_BOOST != 1.0:
                weight = weight * REFINE_BOOST
            repeat = max(1, min(10, round(weight)))
            messages = to_messages(rec)
            # One JSONL row per repeat — SFTTrainer sees each as an independent example
            for _ in range(repeat):
                train_rows.append({
                    "messages": messages,
                    # Keep lightweight metadata for debugging inside the kernel
                    "_trace_id": rec.get("trace_id"),
                    "_scenario_pack": rec.get("scenario_pack"),
                    "_target_label": rec.get("target_label"),
                    "_variant": rec.get("variant", split_name),
                    "_action": rec.get("target_json", {}).get("recommended_action"),
                })
        print(f"  {split_name}: {len(records)} records")

    import random
    random.seed(42)
    random.shuffle(train_rows)

    write_jsonl(OUT_DIR / "simsat_train.jsonl", train_rows)

    # ── Eval splits (keep raw + add messages for the kernel) ─────────────────
    for filename, out_name in [
        ("simsat_eval_reviewed.jsonl", "simsat_eval_reviewed.jsonl"),
        ("simsat_eval_shortlist.jsonl", "simsat_eval_shortlist.jsonl"),
    ]:
        records = load_jsonl(EXPORT_DIR / filename)
        enriched = []
        for rec in records:
            row = dict(rec)
            row["messages"] = to_messages(rec)
            enriched.append(row)
        write_jsonl(OUT_DIR / out_name, enriched)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("Dataset ready for upload:")
    action_counts: dict[str, int] = {}
    for row in train_rows:
        a = row.get("_action", "?")
        action_counts[a] = action_counts.get(a, 0) + 1
    refine_boost_str = f" (REFINE_BOOST={REFINE_BOOST})" if REFINE_BOOST != 1.0 else ""
    print(f"  simsat_train.jsonl:          {len(train_rows)} rows (weighted repeats{refine_boost_str})")
    print(f"  action distribution:         {action_counts}")
    for fname in OUT_DIR.glob("*.jsonl"):
        size = fname.stat().st_size
        print(f"  {fname.name}: {size:,} bytes")

    print()
    print("Next: push with push.py")


if __name__ == "__main__":
    main()
