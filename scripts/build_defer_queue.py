#!/usr/bin/env python3
"""Build a targeted defer-candidate review queue.

The Stage 1 + Stage 2 BC heads (Liquid Track) collapse to 0/75 defer
predictions in eval despite stratified val_acc 1.000 on defer at training.
Root cause: only 3 original defer cases in the 75-trace corpus, and
Stage 2's augmentation overfits to those 3 — synthetic defer features
don't generalize.

The robust fix is more REAL defer reviews, not more augmentation. This
script surfaces traces that are probable defer candidates by probe
signal — high cloud cover with a visible target, or marginal elevation
on a clear scene — and writes them to a queue file the existing
`batch_review.py` understands. The user (operator) reviews each via
the standard a/r/d/s keystroke flow; defer-eligible cases get the
`d` label and grow the corpus.

Usage:
    python scripts/build_defer_queue.py
    python scripts/build_defer_queue.py --out review_queue_defer.txt
    python scripts/build_defer_queue.py --size 20 --cloud-min 50

Then triage with:
    python scripts/batch_review.py --inprocess --reviewer ben \
        --queue-file review_queue_defer.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "review_queue_assets"
TRACES_PATH = REPO_ROOT / "src" / "sim" / "data" / "observation_vla" / "traces.json"
OUTCOMES_PATH = REPO_ROOT / "src" / "sim" / "data" / "observation_vla" / "outcomes.json"


def _has_asset(trace_id: str) -> bool:
    return any(ASSETS_DIR.glob(f"*{trace_id}*.png"))


def _trace_target(trace: dict) -> str:
    sample = trace.get("sample") or {}
    return trace.get("target_id") or sample.get("target_id") or "unknown"


def _is_defer_candidate(trace: dict, *, cloud_min: float, elev_max: float) -> tuple[bool, str]:
    """Heuristic: traces likely worth a 'wait for better window' decision.

    Returns (is_candidate, reason). Reasons:
      cloudy_visible: cloud_cover > cloud_min AND target_visible
      marginal_elev:  elevation < elev_max AND target_visible
    """
    probe = trace.get("probe") or {}
    geom = trace.get("geometry") or {}
    cloud = probe.get("sentinel_cloud_cover")
    elev = geom.get("elevation_degrees")
    visible = geom.get("target_visible") or probe.get("mapbox_feasible")
    if not visible:
        return False, ""
    if cloud is not None and cloud > cloud_min:
        return True, f"cloudy_visible(cloud={cloud:.0f}%)"
    if elev is not None and elev < elev_max:
        return True, f"marginal_elev(elev={elev:.1f}deg)"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "review_queue_defer.txt")
    parser.add_argument("--size", type=int, default=20,
                        help="Max queue size; 0 = include every candidate.")
    parser.add_argument("--per-target-cap", type=int, default=4,
                        help="Max candidates per target so one target doesn't dominate.")
    parser.add_argument("--cloud-min", type=float, default=60.0,
                        help="Min cloud_cover (%) for cloudy_visible candidacy.")
    parser.add_argument("--elev-max", type=float, default=35.0,
                        help="Max elevation (deg) for marginal_elev candidacy.")
    parser.add_argument("--include-already-reviewed", action="store_true",
                        help="By default already-operator-reviewed traces are skipped.")
    args = parser.parse_args()

    traces = json.loads(TRACES_PATH.read_text())["traces"]
    outcomes = json.loads(OUTCOMES_PATH.read_text()).get("outcomes", [])
    op_labels = {
        o["trace_id"]: o["operator_action"]
        for o in outcomes
        if o.get("label_source") == "operator_review"
    }

    candidates: list[dict] = []
    for trace in traces:
        tid = trace.get("trace_id")
        if not tid or not _has_asset(tid):
            continue
        if tid in op_labels and not args.include_already_reviewed:
            continue
        ok, reason = _is_defer_candidate(trace, cloud_min=args.cloud_min, elev_max=args.elev_max)
        if not ok:
            continue
        candidates.append({
            "trace_id": tid,
            "target": _trace_target(trace),
            "scenario_pack": trace.get("scenario_pack", "unknown"),
            "reason": reason,
            "current_label": op_labels.get(tid) or (trace.get("assessment") or {}).get("recommended_action") or "unlabelled",
        })

    # Per-target cap so one busy target doesn't dominate.
    per_target = defaultdict(int)
    chosen: list[dict] = []
    for c in candidates:
        if args.per_target_cap and per_target[c["target"]] >= args.per_target_cap:
            continue
        chosen.append(c)
        per_target[c["target"]] += 1
        if args.size and len(chosen) >= args.size:
            break

    lines: list[str] = [
        "# Defer-candidate review queue (generated by build_defer_queue.py)",
        f"# {len(chosen)} candidates  cloud_min={args.cloud_min}%  elev_max={args.elev_max}deg",
        f"# per_target_cap={args.per_target_cap}  size={args.size}",
        "# Spread by target:",
    ]
    by_target = defaultdict(list)
    for c in chosen:
        by_target[c["target"]].append(c)
    for tgt, items in sorted(by_target.items()):
        sample_summaries = [f"{it['reason']} now={it['current_label']}" for it in items[:3]]
        suffix = "..." if len(items) > 3 else ""
        lines.append(f"#   {tgt}: {len(items)} ({', '.join(sample_summaries)}{suffix})")
    lines.append("# Review with: python scripts/batch_review.py --inprocess --reviewer <you> --queue-file " + args.out.name)
    lines.append("# Lean 'd' (defer) where the imagery is genuinely 'wait for a better pass' worthy.")
    lines.append("")
    lines.extend(c["trace_id"] for c in chosen)

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(chosen)} defer candidates -> {args.out}")
    print(f"Per-target spread:")
    for tgt, items in sorted(by_target.items()):
        print(f"  {tgt:32s}  {len(items)} candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
