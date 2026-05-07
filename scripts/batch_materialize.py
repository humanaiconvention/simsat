#!/usr/bin/env python3
"""Materialize + assess all unreviewed decisions in the encounter pool.

Each decision becomes a new unreviewed trace in the observation-vla pool,
ready for gallery_review.html to pick up.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from operator_review import InProcessClient


def main() -> int:
    c = InProcessClient()
    d = c.get_json("/encounter/decisions")
    decs = d.get("decisions", d) if isinstance(d, dict) else d
    # Filter to decisions whose stimulus is not yet materialized (no trace_id link).
    # The /encounter/decisions response doesn't carry a trace_id, so we just try
    # all of them and let /materialize be idempotent.
    print(f"Decisions on hand: {len(decs)}")

    new_traces: list[str] = []
    failures: list[tuple[str, str]] = []
    for i, dec in enumerate(decs):
        did = dec["decision_id"]
        action = dec.get("action", "?")
        target = dec.get("target_id", "?")
        try:
            t0 = time.time()
            c.post_json(f"/encounter/decision/{did}/materialize")
            res = c.post_json(f"/encounter/decision/{did}/assess")
            tid = res.get("trace_id")
            dur = time.time() - t0
            print(f"[{i+1:>3}/{len(decs)}] {did[:16]}  {action:<6} {target:<28} -> {tid[:16] if tid else 'none'}  ({dur:.1f}s)")
            if tid:
                new_traces.append(tid)
        except Exception as e:
            msg = str(e)[:200]
            print(f"[{i+1:>3}/{len(decs)}] {did[:16]}  FAIL: {msg}")
            failures.append((did, msg))

    print()
    print(f"Created/refreshed {len(new_traces)} traces; {len(failures)} failures.")
    return 0 if new_traces else 1


if __name__ == "__main__":
    sys.exit(main())
