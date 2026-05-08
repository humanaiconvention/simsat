#!/usr/bin/env python3
"""Defer-targeted materialization: candidates the operator might flip to
`defer` rather than `refine`.

Defer ("too compromised, wait for next overpass") sits between refine
(partial cloud, secondary pass) and skip (heavy cloud, not worth bandwidth).
Empirically, the operator's threshold for defer correlates with
60-80% cloud + bad geometry (off_nadir > 30° or elevation < 30°).

Strategy:
  1. Run /encounter/plan at multiple extended horizons (24h-720h).
  2. Materialize ALL fresh decisions (cloud probe runs at materialize).
  3. Filter resulting traces to cloud >= 55% OR (off_nadir > 28° AND
     elevation < 35°) — the "borderline defer" envelope.
  4. Snapshot the candidate set + emit summary.
"""
from __future__ import annotations
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from operator_review import InProcessClient


def _materialize_one(client, did):
    try:
        client.post_json(f"/encounter/decision/{did}/materialize")
        res = client.post_json(f"/encounter/decision/{did}/assess")
        return res.get("trace_id")
    except Exception:
        return None


def _trace_meta(client, tid):
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


def _is_defer_candidate(meta):
    cloud = meta.get("cloud") or 0
    off_nadir = meta.get("off_nadir") or 0
    elev = meta.get("elev") or 90
    # Defer envelope: heavy-but-not-blocking cloud OR bad geometry
    if cloud and 55 <= cloud < 90:
        return True
    if off_nadir and elev and off_nadir > 28 and elev < 35:
        return True
    return False


def main() -> int:
    c = InProcessClient()
    pool_before = c.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    pool_ids_before = {t["trace_id"] for t in pool_before}
    print(f"Pool before: {len(pool_before)}")

    seen = {d["decision_id"] for d in c.get_json("/encounter/decisions").get("decisions", [])}
    print(f"Existing decisions: {len(seen)}")

    new_dids = []
    horizons = [24, 48, 96, 168, 240, 336, 480, 720]
    for h in horizons:
        try:
            plan = c.post_json("/encounter/plan", payload={"horizon_hours": h, "materialize_top_k": 0})
            decs = plan.get("decisions", [])
            new_in_call = 0
            for d in decs:
                did = d["decision_id"]
                if did not in seen:
                    new_dids.append(did)
                    seen.add(did)
                    new_in_call += 1
            print(f"  horizon={h:>4}h: +{new_in_call} new (cumulative: {len(new_dids)})")
        except Exception as e:
            print(f"  horizon={h}: error {e}")

    print(f"\nFresh decisions to materialize: {len(new_dids)}")
    if not new_dids:
        print("Planner saturated — pool expansion exhausted at this horizon range.")
        return 0

    new_tids = []
    for i, did in enumerate(new_dids):
        t0 = time.time()
        tid = _materialize_one(c, did)
        if tid:
            new_tids.append(tid)
        if (i + 1) % 5 == 0 or i + 1 == len(new_dids):
            print(f"  [{i+1:>3}/{len(new_dids)}]  +{len(new_tids)} traces  ({time.time() - t0:.1f}s last)")

    pool_after = c.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    new_traces = [t for t in pool_after if t["trace_id"] not in pool_ids_before]
    print(f"\nNew traces: {len(new_traces)}")

    metas = []
    for t in new_traces:
        try:
            metas.append(_trace_meta(c, t["trace_id"]))
        except Exception:
            pass

    defer_envelope = [m for m in metas if _is_defer_candidate(m)]
    other = [m for m in metas if not _is_defer_candidate(m)]
    print(f"Defer envelope (cloud 55-90% OR off_nadir>28° + elev<35°): {len(defer_envelope)}")
    print(f"Other new: {len(other)}")

    cloud_bins = Counter()
    for m in defer_envelope:
        c_v = m.get("cloud") or 0
        if c_v < 60: cloud_bins["55-60"] += 1
        elif c_v < 70: cloud_bins["60-70"] += 1
        elif c_v < 80: cloud_bins["70-80"] += 1
        else: cloud_bins["80-90"] += 1
    print(f"Defer-envelope cloud distribution: {dict(cloud_bins)}")
    targets = Counter(m.get("target_label") for m in defer_envelope)
    print(f"Targets in envelope: {dict(targets)}")

    out = {
        "summary": {
            "new_traces": len(new_traces),
            "defer_envelope": len(defer_envelope),
            "other": len(other),
        },
        "defer_envelope": defer_envelope,
        "other": other,
    }
    snap = ROOT / "scripts" / "_defer_snapshot.json"
    snap.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSnapshot -> {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
