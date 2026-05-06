#!/usr/bin/env python3
"""Apply JSON reviews from defer_review.html to outcomes.json via InProcessClient.

Usage:
    python scripts/apply_defer_reviews.py reviews.json
    # or pipe JSON directly:
    echo '[{"trace_id":"trace_abc","operator_action":"defer"}]' | python scripts/apply_defer_reviews.py -

The JSON is a list of {trace_id, operator_action} objects — exactly what
defer_review.html outputs via "Copy JSON to clipboard".
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

REVIEWER = "ben"
_SCORES = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: apply_defer_reviews.py <reviews.json | ->")
        sys.exit(1)

    src = sys.argv[1]
    raw = sys.stdin.read() if src == "-" else Path(src).read_text()
    rows: list[dict] = json.loads(raw)
    print(f"Applying {len(rows)} reviews...")

    from operator_review import InProcessClient  # type: ignore
    client = InProcessClient()
    ok = err = 0
    for row in rows:
        tid = row["trace_id"]
        action = row["operator_action"]
        try:
            client.post_json(
                f"/observation-vla/trace/{tid}/operator-review",
                {
                    "reviewer": REVIEWER,
                    "operator_action": action,
                    "useful": action in ("accept", "refine"),
                    "usefulness_score": _SCORES[action],
                    "tags": ["defer_queue_review"],
                },
            )
            print(f"  ✓ {tid[:20]}  {action}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {tid[:20]}  {e}")
            err += 1
    client.close()

    print(f"\nDone: {ok} applied, {err} errors")
    if ok:
        print("Run muzero_stage2_pretrain.py to train v7 with updated corpus.")

if __name__ == "__main__":
    main()
