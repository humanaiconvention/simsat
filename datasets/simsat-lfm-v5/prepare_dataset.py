#!/usr/bin/env python3
"""
Prepare SimSat LFM2.5-VL v2 training dataset for Kaggle upload.

Pulls operator-reviewed traces directly from the live observation-vla API
(SQLite-backed) so the new 51 reviews from gallery_review.html are
included. Then writes the LFM VLM messages format.

Outputs:
  simsat_lfm_train.jsonl       — 4-class stratified train split
  simsat_lfm_holdout.jsonl     — 8 per class held-out (deterministic)
  simsat_lfm_eval.jsonl        — legacy eval rows
  images/                      — copied PNG tiles
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from operator_review import InProcessClient  # type: ignore

OUT_DIR = Path(__file__).parent
OUT_IMAGES = OUT_DIR / "images"

V1_DIR = ROOT / "datasets" / "simsat-lfm-v1"
V1_IMAGES = V1_DIR / "images"
V2_IMAGES = ROOT / "datasets" / "simsat-lfm-v2" / "images"
V3_IMAGES = ROOT / "datasets" / "simsat-lfm-v3" / "images"

EXPORT_DIR = ROOT / "exports" / "gemma4_v4"  # source for prompt_text on existing 145

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

USEFULNESS = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}


def _build_prompt_text(scenario: str, target: str, sample: dict) -> str:
    """Build the user prompt text from trace context."""
    geom = sample.get("geometry") or {}
    probe = sample.get("probe") or {}
    parts = [
        f"Scenario pack: {scenario}",
        f"Target: {target}",
    ]
    elev = geom.get("elevation_degrees")
    if elev is not None:
        parts.append(f"Elevation: {elev:.1f}°")
    nadir = geom.get("off_nadir_degrees")
    if nadir is not None:
        parts.append(f"Off-nadir: {nadir:.1f}°")
    cc = probe.get("sentinel_cloud_cover")
    if cc is not None:
        parts.append(f"Cloud cover: {cc:.1f}%")
    parts.append(
        "Assess this Earth observation for mission usefulness and return the JSON assessment."
    )
    return "\n".join(parts)


def _operator_target_json(action: str, sample: dict) -> dict:
    """Build an ObservationVerdict-shaped JSON from the operator action.

    Uses heuristic score bands by action class plus geometry/cloud signals.
    """
    geom = sample.get("geometry") or {}
    probe = sample.get("probe") or {}
    cloud_raw = probe.get("sentinel_cloud_cover")
    cloud = (cloud_raw if cloud_raw is not None else 10.0) / 100.0
    elev_raw = geom.get("elevation_degrees")
    elev_norm = (elev_raw if elev_raw is not None else 45.0) / 90.0

    if action == "accept":
        base = {"scene_match_score": 0.85, "salience_score": 0.80,
                "change_or_event_score": 0.55, "confidence": 0.85,
                "usable_observation": True}
    elif action == "refine":
        base = {"scene_match_score": 0.65, "salience_score": 0.65,
                "change_or_event_score": 0.50, "confidence": 0.70,
                "usable_observation": True}
    elif action == "defer":
        base = {"scene_match_score": 0.55, "salience_score": 0.55,
                "change_or_event_score": 0.40, "confidence": 0.55,
                "usable_observation": False}
    else:  # skip
        base = {"scene_match_score": 0.30, "salience_score": 0.35,
                "change_or_event_score": 0.20, "confidence": 0.40,
                "usable_observation": False}

    base["occlusion_or_cloud_risk"] = round(min(0.95, max(0.05, cloud + 0.05)), 2)
    base["recommended_action"] = action

    tags = []
    if cloud > 0.6:
        tags.append("heavy_cloud")
    elif cloud > 0.3:
        tags.append("moderate_cloud")
    else:
        tags.append("low_cloud")
    if elev_norm < 0.3:
        tags.append("low_elevation")
    if action == "accept":
        tags.append("high_priority")
    if action == "skip":
        tags.append("not_useful")
    base["rationale_tags"] = tags
    return base


def _fetch_image_for_trace(client: InProcessClient, sample: dict, images_meta: list) -> Optional[bytes]:
    """Fetch a 15km Sentinel-2 RGB tile via the api."""
    if not images_meta:
        return None
    md = images_meta[0].get("metadata", {})
    target_meta = sample.get("target_metadata") or {}
    lon = sample.get("target_lon") or target_meta.get("lon")
    lat = sample.get("target_lat") or target_meta.get("lat")
    if lon is None or lat is None:
        fp = md.get("footprint")
        if fp and len(fp) == 4:
            lon = (fp[0] + fp[2]) / 2
            lat = (fp[1] + fp[3]) / 2
    size_km = md.get("size_km") or 15.0
    ts = md.get("timestamp") or images_meta[0].get("timestamp")
    if lon is None or lat is None or ts is None:
        return None
    res = client._context.get(
        "/data/image/sentinel",
        params={"lon": lon, "lat": lat, "size_km": size_km, "timestamp": ts},
    )
    if res.status_code != 200:
        return None
    return res.content


def main() -> int:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    client = InProcessClient()
    pool = client.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    print(f"Live pool: {len(pool)} traces")

    # ---- Pull all operator-reviewed traces ----
    rows: list[dict] = []
    cache_v1 = {p.name: p for p in V1_IMAGES.iterdir()} if V1_IMAGES.exists() else {}
    # v2 has the 50+20 materialized images on top of v1
    cache_v2 = {p.name: p for p in V2_IMAGES.iterdir()} if V2_IMAGES.exists() else {}
    # v3 + (latest from defer materializer)
    cache_v3 = {p.name: p for p in V3_IMAGES.iterdir()} if V3_IMAGES.exists() else {}
    # newest wins
    cache_v1.update(cache_v2)
    cache_v1.update(cache_v3)

    for i, t in enumerate(pool):
        tid = t["trace_id"]
        detail = client.get_json(f"/observation-vla/trace/{tid}")
        cur = detail.get("current_outcome") or {}
        if cur.get("label_source") != "operator_review":
            continue

        op_action = cur.get("operator_action") or cur.get("recommended_action") or cur.get("action")
        if op_action not in {"accept", "refine", "defer", "skip"}:
            continue

        trace = detail["trace"]
        sample = trace.get("sample") or {}
        scenario = trace.get("scenario_pack") or sample.get("scenario_pack") or "?"
        target = sample.get("target_label") or sample.get("target_id") or "?"
        target_id = sample.get("target_id") or "?"

        # Image source: prefer existing v1 file (already cached), else fetch
        # from /data/image/sentinel.
        png_name = f"trace_{tid.replace('trace_', '')}.png"
        out_path = OUT_IMAGES / png_name
        # If existing file is zero/empty, force re-fetch
        if out_path.exists() and out_path.stat().st_size < 1024:
            out_path.unlink()
        if out_path.exists():
            pass  # already there
        elif png_name in cache_v1 and cache_v1[png_name].stat().st_size >= 1024:
            shutil.copy(cache_v1[png_name], out_path)
        else:
            content = _fetch_image_for_trace(client, sample, sample.get("images") or [])
            if content is None or len(content) < 1024:
                print(f"  [{i+1}] {tid[:24]}  no image / empty content; skipping row")
                continue
            out_path.write_bytes(content)
            time.sleep(2.5)  # polite throttle to stay under 30/min

        # Validate the image is decodable before including the row.
        try:
            from PIL import Image as _PIL
            _im = _PIL.open(out_path)
            _im.load()
        except Exception as e:
            print(f"  [{i+1}] {tid[:24]}  bad image ({e}); deleting + skipping row")
            try:
                out_path.unlink()
            except Exception:
                pass
            continue

        target_json = _operator_target_json(op_action, sample)
        prompt_text = _build_prompt_text(scenario, target, sample)
        rec = {
            "trace_id": tid,
            "scenario_pack": scenario,
            "target_id": target_id,
            "target_label": target,
            "image_path": f"images/{png_name}",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"images/{png_name}"},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(target_json)}],
                },
            ],
        }
        rows.append(rec)

    print(f"\nTotal reviewed rows ready: {len(rows)}")
    by_action = Counter(r["messages"][-1]["content"][0]["text"] and json.loads(r["messages"][-1]["content"][0]["text"])["recommended_action"] for r in rows)
    print(f"Action distribution: {dict(by_action)}")

    # ---- Stratified split: 8 per class hold-out, deterministic ----
    import random
    rng = random.Random(42)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        a = json.loads(r["messages"][-1]["content"][0]["text"])["recommended_action"]
        by_class[a].append(r)

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    HOLDOUT_PER_CLASS = 8
    for a, lst in by_class.items():
        lst_sorted = sorted(lst, key=lambda r: r["trace_id"])
        rng.shuffle(lst_sorted)
        if len(lst_sorted) <= HOLDOUT_PER_CLASS:
            train_rows.extend(lst_sorted)
        else:
            holdout_rows.extend(lst_sorted[:HOLDOUT_PER_CLASS])
            train_rows.extend(lst_sorted[HOLDOUT_PER_CLASS:])

    # Deterministic: keep holdout = same trace_ids as v1 if present (so direct
    # base/tuned comparison stays valid).
    v1_holdout_path = V1_DIR / "simsat_lfm_holdout.jsonl"
    if v1_holdout_path.exists():
        v1_holdout_ids = {
            json.loads(line)["trace_id"]
            for line in v1_holdout_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        # Pull v1 holdout ids out of train if they ended up there; force them
        # back into holdout for apples-to-apples comparison.
        new_train = [r for r in train_rows if r["trace_id"] not in v1_holdout_ids]
        forced_holdout = [r for r in train_rows if r["trace_id"] in v1_holdout_ids]
        # Drop any holdout entry not in v1 so the holdout stays identical to v1
        new_holdout = [r for r in holdout_rows if r["trace_id"] in v1_holdout_ids] + forced_holdout
        # Pull any v1 holdout ids that ended up missing from rows
        existing_holdout_ids = {r["trace_id"] for r in new_holdout}
        # Whatever is left in old holdout but not v1 should go to train
        leftover = [r for r in holdout_rows if r["trace_id"] not in v1_holdout_ids]
        new_train.extend(leftover)
        train_rows = new_train
        holdout_rows = new_holdout
        print(f"\nForced holdout to match v1 ({len(holdout_rows)} rows)")

    # Write outputs
    def write_jsonl(path: Path, rows: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote {len(rows)} rows -> {path.name}")

    write_jsonl(OUT_DIR / "simsat_lfm_train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "simsat_lfm_holdout.jsonl", holdout_rows)

    # Eval rows: copy from v1 if exists
    v1_eval = V1_DIR / "simsat_lfm_eval.jsonl"
    if v1_eval.exists():
        shutil.copy(v1_eval, OUT_DIR / "simsat_lfm_eval.jsonl")
        n = sum(1 for _ in open(OUT_DIR / "simsat_lfm_eval.jsonl", encoding="utf-8"))
        print(f"  Copied legacy eval ({n} rows)")

    # Summary
    print("\nTrain action distribution:")
    train_actions = Counter(json.loads(r["messages"][-1]["content"][0]["text"])["recommended_action"] for r in train_rows)
    for a, n in sorted(train_actions.items()):
        print(f"  {a:<8} {n}")
    print("Holdout action distribution:")
    hold_actions = Counter(json.loads(r["messages"][-1]["content"][0]["text"])["recommended_action"] for r in holdout_rows)
    for a, n in sorted(hold_actions.items()):
        print(f"  {a:<8} {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
