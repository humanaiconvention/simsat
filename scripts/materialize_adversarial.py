#!/usr/bin/env python3
"""Materialize adversarial encounters: high-cloud and bad-geometry windows
that the standard planner would normally rank low.

Strategy:
  1. For each known target, request windows over a long horizon (~336 h).
  2. Pick windows whose geometry is marginal (off_nadir > 30° OR
     elevation < 35°). These are the ones the analytic policy would push
     down the rank list.
  3. Force a /encounter/plan call to generate decisions for those windows
     (the planner will accept them but mark trust_band=low/medium).
  4. Materialize + assess each — the cloud probe runs at materialize time,
     so only afterwards do we know the cloud cover.
  5. Filter materialized traces to those with cloud > 50% OR off_nadir > 30°.
     These are our adversarial candidates ready for operator review.

The existing operator-reviewed pool already has 51 traces at >80% cloud
(verified by inspection); this script grows that count further by
materializing candidates the planner skipped over.
"""
from __future__ import annotations
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from operator_review import InProcessClient


THUMB_DIR = Path("/tmp/gallery_thumbs")
THUMB_DIR.mkdir(parents=True, exist_ok=True)


def _materialize_one(client: InProcessClient, decision_id: str) -> str | None:
    try:
        client.post_json(f"/encounter/decision/{decision_id}/materialize")
        res = client.post_json(f"/encounter/decision/{decision_id}/assess")
        return res.get("trace_id")
    except Exception:
        return None


def _trace_meta(client: InProcessClient, tid: str) -> dict:
    detail = client.get_json(f"/observation-vla/trace/{tid}")
    sample = detail["trace"]["sample"]
    images = sample.get("images") or []
    md = images[0].get("metadata", {}) if images else {}
    probe = sample.get("probe") or {}
    geom = sample.get("geometry") or {}
    return {
        "trace_id": tid,
        "scenario": detail["trace"].get("scenario_pack"),
        "target_label": sample.get("target_label"),
        "cloud": probe.get("sentinel_cloud_cover"),
        "elev": geom.get("elevation_degrees"),
        "off_nadir": geom.get("off_nadir_degrees"),
    }


def main() -> int:
    c = InProcessClient()

    # Step 1: existing pool
    pool_before = c.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    pool_ids_before = {t["trace_id"] for t in pool_before}
    print(f"Pool size before: {len(pool_before)}")

    # Step 2: collect candidate decisions across multiple plan calls.
    # The planner only emits a few per call — but we can vary the horizon
    # to explore different orbital windows and get fresh candidates.
    new_decision_ids: list[str] = []
    seen = {d["decision_id"] for d in c.get_json("/encounter/decisions").get("decisions", [])}
    print(f"Existing decisions on hand: {len(seen)}")

    for horizon in [12, 36, 72, 168, 336]:
        try:
            plan = c.post_json(
                "/encounter/plan",
                payload={"horizon_hours": horizon, "materialize_top_k": 0,
                         "max_decisions": 100},
            )
        except Exception:
            plan = c.post_json(
                "/encounter/plan",
                payload={"horizon_hours": horizon, "materialize_top_k": 0},
            )
        decisions = plan.get("decisions", [])
        new_in_call = 0
        for d in decisions:
            did = d["decision_id"]
            if did not in seen:
                new_decision_ids.append(did)
                seen.add(did)
                new_in_call += 1
        print(f"  horizon={horizon}h plan: {len(decisions)} returned, +{new_in_call} new")

    print(f"\nNew decisions to materialize: {len(new_decision_ids)}")
    if not new_decision_ids:
        print("No fresh decisions from planner — pool already saturated.")
        # Fall back: re-materialize existing decisions to get diversity
        all_decs = c.get_json("/encounter/decisions").get("decisions", [])
        print(f"Falling back to re-materializing {len(all_decs)} existing decisions")
        new_decision_ids = [d["decision_id"] for d in all_decs]

    # Step 3: materialize all candidates
    new_trace_ids: list[str] = []
    for i, did in enumerate(new_decision_ids):
        t0 = time.time()
        tid = _materialize_one(c, did)
        dur = time.time() - t0
        if tid:
            new_trace_ids.append(tid)
        if (i + 1) % 5 == 0 or i + 1 == len(new_decision_ids):
            print(f"  [{i+1:>3}/{len(new_decision_ids)}]  materialized={len(new_trace_ids)}  ({dur:.1f}s last)")

    # Step 4: re-pull pool, find new traces
    pool_after = c.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    new_traces = [t for t in pool_after if t["trace_id"] not in pool_ids_before]
    print(f"\nNew traces in pool: {len(new_traces)}")

    # Step 5: gather metadata + filter for adversarial conditions
    adversarial: list[dict] = []
    other_unreviewed: list[dict] = []
    for t in new_traces:
        tid = t["trace_id"]
        try:
            meta = _trace_meta(c, tid)
            cloud = meta.get("cloud") or 0
            off_nadir = meta.get("off_nadir") or 0
            elev = meta.get("elev") or 90
            is_adversarial = (cloud and cloud >= 50) or (off_nadir and off_nadir >= 30) or (elev and elev <= 35)
            if is_adversarial:
                adversarial.append(meta)
            else:
                other_unreviewed.append(meta)
        except Exception as e:
            print(f"  meta error for {tid[:24]}: {e}")

    print(f"\nAdversarial (cloud>=50% OR off_nadir>=30° OR elev<=35°): {len(adversarial)}")
    print(f"Non-adversarial new traces: {len(other_unreviewed)}")
    # Cloud distribution among adversarial
    cloud_bins = Counter()
    for m in adversarial:
        c_v = m.get("cloud") or 0
        if c_v < 20: cloud_bins["<20"] += 1
        elif c_v < 50: cloud_bins["20-50"] += 1
        elif c_v < 80: cloud_bins["50-80"] += 1
        else: cloud_bins[">=80"] += 1
    print(f"Adversarial cloud distribution: {dict(cloud_bins)}")
    targets = Counter(m.get("target_label") for m in adversarial)
    print(f"Adversarial targets: {dict(targets)}")

    # Persist a snapshot
    out = {
        "summary": {
            "new_traces": len(new_traces),
            "adversarial": len(adversarial),
            "other": len(other_unreviewed),
        },
        "adversarial": adversarial,
        "other": other_unreviewed,
    }
    snapshot_path = ROOT / "scripts" / "_adversarial_snapshot.json"
    snapshot_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSnapshot -> {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
