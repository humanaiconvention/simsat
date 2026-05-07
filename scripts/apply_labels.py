#!/usr/bin/env python3
"""Apply labels from a JSON file (downloaded from gallery_review.html) to the
SimSat operator-review API.

The gallery HTML produces a JSON file like:
    {
        "labels": [
            {"trace_id": "trace_xxx", "action": "accept"},
            ...
        ],
        "reviewer": "ben"
    }

This script POSTs each one to /observation-vla/trace/{id}/operator-review.

Usage:
    python scripts/apply_labels.py path/to/labels.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from operator_review import InProcessClient  # type: ignore


USEFUL_FLAG = {"accept": True, "refine": True, "defer": False, "skip": False}
USEFULNESS = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_file", type=Path, help="Path to labels.json from gallery_review.html")
    parser.add_argument("--reviewer", default=None, help="Override reviewer name (default: from JSON)")
    parser.add_argument("--mission-id", default="default")
    args = parser.parse_args()

    if not args.labels_file.exists():
        print(f"ERROR: labels file not found: {args.labels_file}")
        sys.exit(1)

    payload = json.loads(args.labels_file.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    reviewer = args.reviewer or payload.get("reviewer") or "ben"
    if not labels:
        print("ERROR: no labels in file")
        sys.exit(1)

    print(f"Applying {len(labels)} labels (reviewer={reviewer})")
    client = InProcessClient()
    success = 0
    fail = 0
    fail_traces = []
    for i, item in enumerate(labels, start=1):
        tid = item.get("trace_id")
        action = item.get("action")
        if not tid or action not in USEFUL_FLAG:
            print(f"  [{i}/{len(labels)}] skip — invalid: {item}")
            fail += 1
            continue
        try:
            client.post_json(
                f"/observation-vla/trace/{tid}/operator-review",
                {
                    "operator_action": action,
                    "reviewer": reviewer,
                    "useful": USEFUL_FLAG[action],
                    "usefulness_score": USEFULNESS[action],
                    "mission_id": args.mission_id,
                },
            )
            success += 1
            if i % 25 == 0:
                print(f"  [{i}/{len(labels)}] {success} ok, {fail} fail")
        except Exception as e:
            fail += 1
            fail_traces.append((tid, str(e)[:100]))
            print(f"  [{i}/{len(labels)}] FAIL {tid[:30]}: {str(e)[:120]}")

    print(f"\nDone. Success: {success}, Fail: {fail}")
    if fail_traces:
        print("Failures:")
        for tid, err in fail_traces[:10]:
            print(f"  {tid}: {err}")


if __name__ == "__main__":
    main()
