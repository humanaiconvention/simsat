#!/usr/bin/env python3
"""Fast keystroke-driven batch labeling for ObservationVLA traces.

The standing operator_review.py CLI handles one trace per invocation with
prompts for every field, which makes ground-truth expansion (the bottleneck
for v11+ training) painfully slow. This wrapper loops through candidate
traces, opens each image asset in the OS image viewer, and accepts a single
keystroke per case (a/r/d/s/n/q). All other fields default sensibly.

Usage:
    python scripts/batch_review.py --inprocess --limit 30 --reviewer ben

Then per case, hit one of:
    a  accept  (useful=true,  usefulness=0.85)
    r  refine  (useful=true,  usefulness=0.55)
    d  defer   (useful=false, usefulness=0.40)
    s  skip    (useful=false, usefulness=0.20)
    n  next    (skip this trace, don't label)
    q  quit    (save progress, exit)

A --queue-file <path> loads a pre-curated newline-delimited list of trace_ids
to label in order; otherwise candidates come from the standing
list_candidates() shortlist (one best pending per target).
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from operator_review import HttpClient, InProcessClient, list_candidates, show_review_bundle  # type: ignore


# action -> (useful, usefulness_score) — sensible defaults; reviewer can override later.
_ACTION_DEFAULTS = {
    "accept": (True,  0.85),
    "refine": (True,  0.55),
    "defer":  (False, 0.40),
    "skip":   (False, 0.20),
}
_KEY_TO_ACTION = {"a": "accept", "r": "refine", "d": "defer", "s": "skip"}


def _open_in_viewer(path: Path) -> None:
    """Best-effort cross-platform image preview."""
    if not path.exists():
        print(f"  [no asset on disk: {path.name}]")
        return
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"  [could not open viewer: {exc}]  asset={path}")


def _resolve_asset(trace_id: str) -> Path:
    """The export pipeline names assets <scenario>_<trace_id>_<suffix>.png."""
    assets_dir = REPO_ROOT / "review_queue_assets"
    matches = sorted(assets_dir.glob(f"*{trace_id}*.png"))
    return matches[0] if matches else assets_dir / f"<missing>_{trace_id}.png"


def _post_review(client, trace_id: str, action: str, reviewer: str, mission_id: str | None, tags: list[str]) -> dict:
    useful, score = _ACTION_DEFAULTS[action]
    payload = client.post_json(
        f"/observation-vla/trace/{trace_id}/operator-review",
        {
            "reviewer": reviewer,
            "operator_action": action,
            "useful": useful,
            "usefulness_score": score,
            "outcome_tags": tags,
            "notes": f"batch_review: keystroke={action[0]}",
            "mission_id": mission_id,
        },
    )
    return payload.get("outcome", {})


def _load_queue(queue_file: Path | None, client, limit: int, scenario_pack: str | None) -> list[str]:
    if queue_file is not None:
        return [
            line.strip()
            for line in queue_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    candidates = list_candidates(client, scenario_pack=scenario_pack, limit=limit)
    return [c.get("trace_id") for c in candidates if c.get("trace_id")]


def _read_action(prompt: str) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            continue
        ch = raw[0]
        if ch in _KEY_TO_ACTION or ch in {"n", "q"}:
            return ch
        print("  expected one of: a r d s n q")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app.")
    parser.add_argument("--scenario-pack", default=None)
    parser.add_argument("--limit", type=int, default=20, help="Candidates to fetch when no --queue-file given.")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--mission-id", default=None)
    parser.add_argument("--tag", action="append", default=[], dest="tags")
    parser.add_argument("--queue-file", type=Path, default=None,
                        help="Newline-delimited file of trace_ids to label in order.")
    parser.add_argument("--no-viewer", action="store_true", help="Don't auto-open image assets.")
    parser.add_argument("--no-bundle", action="store_true", help="Don't print review bundle metadata.")
    args = parser.parse_args()

    client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
    try:
        queue = _load_queue(args.queue_file, client, args.limit, args.scenario_pack)
        if not queue:
            print("No candidates to label. (Empty queue file or no pending traces.)")
            return 0

        print(f"Batch review session: {len(queue)} candidates  reviewer={args.reviewer}")
        print("Keys: a=accept  r=refine  d=defer  s=skip  n=next(skip)  q=quit\n")

        labelled = 0
        skipped = 0
        for idx, trace_id in enumerate(queue, start=1):
            asset = _resolve_asset(trace_id)
            print(f"[{idx}/{len(queue)}] trace={trace_id}")
            print(f"   asset={asset.name if asset.exists() else '<missing>'}")
            if not args.no_bundle:
                try:
                    show_review_bundle(client, trace_id)
                except Exception as exc:
                    print(f"   [bundle fetch failed: {exc}]")
            if not args.no_viewer:
                _open_in_viewer(asset)
            ch = _read_action("   action [a/r/d/s/n/q]: ")
            if ch == "q":
                print(f"\nQuit. Labelled {labelled}, skipped {skipped}, remaining {len(queue) - idx}.")
                break
            if ch == "n":
                skipped += 1
                continue
            action = _KEY_TO_ACTION[ch]
            try:
                outcome = _post_review(client, trace_id, action, args.reviewer, args.mission_id, args.tags)
                print(f"   ✓ {action}  outcome_id={outcome.get('outcome_id', '?')[:24]}")
                labelled += 1
            except Exception as exc:
                print(f"   ✗ FAILED to post review: {exc}")
        else:
            print(f"\nDone. Labelled {labelled}, skipped {skipped}.")
        return 0
    finally:
        if hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    sys.exit(main())
