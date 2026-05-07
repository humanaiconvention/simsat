#!/usr/bin/env python3
"""Export all operator-reviewed traces as a JSONL eval bundle for off-machine evaluation.

Output JSONL row schema:
    {
        "trace_id": "trace_xxx",
        "scenario_pack": "...",
        "target_id": "...",
        "target_label": "...",
        "operator_action": "accept|refine|defer|skip",
        "operator_useful": true|false,
        "operator_usefulness": 0.85,
        "reviewer": "ben",
        "prompt": "<system + user prompt for v11 inference>",
        "image_path": "<rel path under review_queue_assets/, if any>",
        "metadata": {
            "cloud_cover_pct": 12.3,
            "elevation_deg": 67.2,
            "off_nadir_deg": 14.5,
            "target_visible": true,
            "sentinel_datetime": "2026-04-09T...",
            "sentinel_source": "sentinel-2b"
        }
    }

Usage:
    python scripts/export_eval_data.py --output simsat_eval_pool.jsonl

Optional bundling:
    python scripts/export_eval_data.py --output simsat_eval_pool.jsonl --bundle simsat_eval_pool.tar.gz
        — also tars the JSONL together with the matching review_queue_assets/ PNGs.
"""
from __future__ import annotations
import argparse
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from operator_review import InProcessClient  # type: ignore


SYSTEM_PROMPT = (
    "You are a satellite encounter-assessment AI. "
    "Given metadata about a planned Earth observation, return a compact JSON assessment. "
    "Fields: usable_observation (bool), scene_match_score (float 0-1), "
    "salience_score (float 0-1), change_or_event_score (float 0-1), "
    "occlusion_or_cloud_risk (float 0-1), confidence (float 0-1), "
    "recommended_action (accept|defer|refine|skip), rationale_tags (list[str]). "
    "Return only valid JSON, no markdown, no explanation."
)


def build_user_prompt(trace_detail: dict) -> str:
    """Reconstruct the canonical user-prompt from trace metadata."""
    trace = trace_detail.get("trace", {})
    sample = trace.get("sample", {})
    probe = sample.get("probe", {})
    geom = sample.get("geometry", {})

    target_label = sample.get("target_label", "?")
    target_tags = sample.get("target_tags", []) or []
    scenario = sample.get("scenario_pack", "?")

    cloud = probe.get("sentinel_cloud_cover") or 0.0
    sent_dt = probe.get("sentinel_datetime") or "?"
    sent_src = probe.get("sentinel_source") or "?"

    elev = geom.get("elevation_degrees") or 0.0
    off_nadir = geom.get("off_nadir_degrees") or 0.0
    visible = geom.get("target_visible", False)

    parts = [
        f"Target: {target_label} ({scenario})",
        f"Tags: {', '.join(target_tags) if target_tags else 'none'}",
        f"Sentinel pass at {sent_dt} ({sent_src})",
        f"Cloud cover: {float(cloud):.1f}%",
        f"Elevation: {float(elev):.1f}°, off-nadir: {float(off_nadir):.1f}°",
        f"Target visible: {visible}",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="simsat_eval_pool.jsonl", type=Path)
    parser.add_argument("--bundle", default=None, type=Path, help="Optional .tar.gz to bundle JSONL + thumbnails")
    parser.add_argument("--reviewed-only", action="store_true", default=True,
                        help="Only export operator_review labels (default: True)")
    parser.add_argument("--include-images", action="store_true",
                        help="Embed image paths (relative to repo root)")
    args = parser.parse_args()

    client = InProcessClient()
    data = client.get_json("/observation-vla/traces", limit=1000)
    traces = data.get("traces", [])

    rows = []
    skipped_no_label = 0
    asset_dir = REPO_ROOT / "review_queue_assets"

    for t in traces:
        tid = t.get("trace_id")
        if not tid:
            continue
        d = client.get_json(f"/observation-vla/trace/{tid}")
        outcome = d.get("current_outcome") or {}
        if args.reviewed_only and outcome.get("label_source") != "operator_review":
            skipped_no_label += 1
            continue

        trace = d.get("trace", {})
        sample = trace.get("sample", {})
        probe = sample.get("probe", {})
        geom = sample.get("geometry", {})

        # Find image
        img_matches = list(asset_dir.glob(f"*{tid}*.png"))
        img_rel = str(img_matches[0].relative_to(REPO_ROOT)) if img_matches else None

        rows.append({
            "trace_id": tid,
            "scenario_pack": sample.get("scenario_pack") or trace.get("scenario_pack"),
            "target_id": sample.get("target_id") or trace.get("target_id"),
            "target_label": sample.get("target_label", "?"),
            "operator_action": outcome.get("operator_action"),
            "operator_useful": outcome.get("useful"),
            "operator_usefulness": outcome.get("usefulness_score"),
            "reviewer": outcome.get("reviewer", "?"),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_user_prompt(d),
            "image_path": img_rel,
            "metadata": {
                "cloud_cover_pct": probe.get("sentinel_cloud_cover"),
                "elevation_deg": geom.get("elevation_degrees"),
                "off_nadir_deg": geom.get("off_nadir_degrees"),
                "target_visible": geom.get("target_visible"),
                "sentinel_datetime": probe.get("sentinel_datetime"),
                "sentinel_source": probe.get("sentinel_source"),
            },
        })

    # Write JSONL
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} reviewed traces to {args.output} (skipped {skipped_no_label} unlabeled)")

    # By-scenario breakdown
    from collections import Counter
    by_s = Counter(r["scenario_pack"] for r in rows)
    by_a = Counter(r["operator_action"] for r in rows)
    print(f"  By scenario: {dict(by_s)}")
    print(f"  By operator action: {dict(by_a)}")

    if args.bundle:
        with tarfile.open(args.bundle, "w:gz") as tar:
            tar.add(args.output, arcname=args.output.name)
            for r in rows:
                if r.get("image_path"):
                    p = REPO_ROOT / r["image_path"]
                    if p.exists():
                        tar.add(p, arcname=r["image_path"])
        size_mb = args.bundle.stat().st_size / 1024 / 1024
        print(f"\nBundle: {args.bundle} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
