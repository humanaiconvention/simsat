#!/usr/bin/env python3
"""
Prepare SimSat LFM2.5-VL v1 training dataset for Kaggle upload.

Reads from <repo_root>/exports/gemma4_v4/ (which contains the same multimodal
operator-reviewed traces used for the Gemma-4 v11 fine-tune) and rewrites them
into the LFM VLM messages format expected by `processor.apply_chat_template`:

    {"messages": [
        {"role": "system",    "content": [{"type": "text",  "text": "..."}]},
        {"role": "user",      "content": [{"type": "image", "image": "<path>"},
                                          {"type": "text",  "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text",  "text": "<json>"}]},
    ]}

Output to this directory:
  simsat_lfm_train.jsonl   — multimodal_reviewed rows (image + reviewed JSON)
  simsat_lfm_eval.jsonl    — eval_reviewed held-out rows
  images/                  — copied PNG tiles (paths in JSONL are relative)

The image paths in `simsat_lfm_*.jsonl` are written as `images/<basename>.png`
so they resolve correctly when the dataset is mounted at `/kaggle/input/simsat-lfm-v1/`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parents[2] / "exports" / "gemma4_v4"
OUT_DIR = Path(__file__).parent
OUT_IMAGES = OUT_DIR / "images"

SYSTEM_PROMPT = (
    "You are a satellite encounter-assessment AI. "
    "Given a Sentinel-2 satellite image and accompanying metadata, return a "
    "compact JSON assessment with these fields exactly: "
    "usable_observation (bool), scene_match_score (float 0-1), "
    "salience_score (float 0-1), change_or_event_score (float 0-1), "
    "occlusion_or_cloud_risk (float 0-1), confidence (float 0-1), "
    "recommended_action (one of: accept, defer, refine, skip), "
    "rationale_tags (list of short strings). "
    "Return only valid JSON, no markdown fences, no explanation."
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(rows)} rows -> {path.name}")


def to_messages(record: dict) -> dict | None:
    """Convert a multimodal export row to LFM VLM messages format.

    Returns None if the row is missing the image or the target.
    """
    if not record.get("has_image"):
        return None
    image_rel = record.get("image_path")  # e.g. "images/trace_xxx.png"
    if not image_rel:
        return None

    target = record.get("target_json")
    if not isinstance(target, dict):
        return None

    # On Kaggle the dataset mounts at /kaggle/input/simsat-lfm-v1/
    # The image_path stored in the JSONL stays relative — the training
    # collate_fn resolves it against the dataset root.
    image_basename = Path(image_rel).name
    kaggle_image_path = f"images/{image_basename}"

    # User content combines the image with the existing prompt_text (which
    # already includes scenario_pack, target, geometry, probe metadata).
    user_text = record.get("prompt_text", "Assess this Earth observation for mission usefulness.")

    return {
        "trace_id": record.get("trace_id"),
        "scenario_pack": record.get("scenario_pack"),
        "target_id": record.get("target_id"),
        "target_label": record.get("target_label"),
        "image_path": kaggle_image_path,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": kaggle_image_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(target)}],
            },
        ],
    }


def _stratified_split(
    rows: list[dict],
    holdout_per_class: int = 8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split rows by recommended_action; reserve `holdout_per_class` rows from
    each action class for evaluation. Deterministic (sorted + seeded).

    The hold-out exists because the existing simsat_eval_reviewed.jsonl has
    only `defer`+`refine` cases — not enough to measure a 4-class fine-tune
    against the base model.
    """
    import random

    by_class: dict[str, list[dict]] = {}
    for r in rows:
        target = r.get("messages", [{}])[2]["content"][0]["text"]
        try:
            act = json.loads(target)["recommended_action"]
        except Exception:
            act = "unknown"
        by_class.setdefault(act, []).append(r)

    rng = random.Random(seed)
    train: list[dict] = []
    holdout: list[dict] = []
    for cls, items in sorted(by_class.items()):
        sorted_items = sorted(items, key=lambda r: r.get("trace_id", ""))
        rng.shuffle(sorted_items)
        n_hold = min(holdout_per_class, len(sorted_items))
        holdout.extend(sorted_items[:n_hold])
        train.extend(sorted_items[n_hold:])
    return train, holdout


def main() -> None:
    print("SimSat LFM-VL v1 dataset preparation")
    print(f"  Source: {EXPORT_DIR}")
    print(f"  Output: {OUT_DIR}")
    print()

    if not EXPORT_DIR.exists():
        raise SystemExit(
            f"Export dir {EXPORT_DIR} not found. Run the upstream gemma4_v4 export first "
            "or symlink an alternative source."
        )

    # ---- Multimodal reviewed pool: stratified split ----------------------
    multimodal_src = load_jsonl(EXPORT_DIR / "simsat_multimodal_reviewed.jsonl")
    multimodal_msgs = [m for m in (to_messages(r) for r in multimodal_src) if m is not None]
    print(f"  ({len(multimodal_src) - len(multimodal_msgs)} rows skipped — missing image or target_json)")

    train_msgs, holdout_msgs = _stratified_split(multimodal_msgs, holdout_per_class=8, seed=42)
    write_jsonl(OUT_DIR / "simsat_lfm_train.jsonl", train_msgs)
    write_jsonl(OUT_DIR / "simsat_lfm_holdout.jsonl", holdout_msgs)

    # ---- Legacy eval set kept for back-compat (defer+refine only) --------
    eval_src = load_jsonl(EXPORT_DIR / "simsat_eval_reviewed.jsonl")
    eval_msgs = [m for m in (to_messages(r) for r in eval_src) if m is not None]
    write_jsonl(OUT_DIR / "simsat_lfm_eval.jsonl", eval_msgs)

    # ---- Copy images into the dataset dir --------------------------------
    OUT_IMAGES.mkdir(exist_ok=True)
    used_images: set[str] = set()
    for m in train_msgs + holdout_msgs + eval_msgs:
        used_images.add(Path(m["image_path"]).name)

    src_images = EXPORT_DIR / "images"
    copied = 0
    missing = []
    for name in sorted(used_images):
        src = src_images / name
        dst = OUT_IMAGES / name
        if not src.exists():
            missing.append(name)
            continue
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
            copied += 1
    print(f"  Images: {len(used_images)} referenced, {copied} newly copied, {len(missing)} missing")
    if missing:
        print(f"  WARNING: missing image basenames: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # ---- Per-class action distribution (sanity) --------------------------
    def _dist(rows: list[dict]) -> dict:
        d: dict[str, int] = {}
        for m in rows:
            try:
                t = json.loads(m["messages"][2]["content"][0]["text"])
                a = t.get("recommended_action", "?")
                d[a] = d.get(a, 0) + 1
            except Exception:
                pass
        return d

    print(f"  Train action distribution:   {_dist(train_msgs)}")
    print(f"  Holdout action distribution: {_dist(holdout_msgs)}")
    print(f"  Eval-legacy action distribution: {_dist(eval_msgs)}")

    print()
    print("Next:")
    print("  python notebooks/kaggle-simsat-lfm-v1/push.py --dataset-only")


if __name__ == "__main__":
    main()
