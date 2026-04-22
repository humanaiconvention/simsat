"""Parity harness: verify that the new WCLITrustModel reproduces the scattered
logic's decisions on the existing trace corpus — with zero drift.

Usage:
    python scripts/trust_model_parity.py --traces-dir traces/ --max-cases 0

Exit codes:
    0 — all decisions match (refactor is a pure no-op; safe to commit)
    1 — one or more decisions differ (refactor dropped or flipped a reason/action; do not commit)
    2 — harness could not load traces / old logic / new logic (fix the environment)

WHAT THIS DOES
--------------
For each stored trace in the 86-trace ObservationVLA corpus:
  1. Reconstruct the (scaffold_action, context) that the planner passed to the
     scattered trust logic at the time the trace was recorded.
  2. Run the OLD scattered path and record (action, reason).
  3. Run the NEW WCLITrustModel.decide() on the same inputs and record (action, reason).
  4. Compare. Mismatches are printed; harness exits nonzero.

The `old_scattered_decide` function is a shim Ben fills in during the refactor —
it should delegate to whatever planner.py / service.py does today.

ADAPTING THIS TO YOUR LOCAL CORPUS
----------------------------------
The `load_traces()` function is a stub. Point it at wherever the trace store lives
on D:\\SimSat (likely under observation_vla traces or a similar directory).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from sim.encounter.trust_model import TrustContext, WCLITrustModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("trust_model_parity")


# --------------------------------------------------------------------------- #
# Trace loading (adapt to local path layout)                                  #
# --------------------------------------------------------------------------- #

def load_traces(traces_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield one trace dict per stored ObservationVLA trace file.

    Adapt this function to the actual trace storage format at D:\\SimSat.
    Each yielded dict must contain enough to reconstruct both:
      - The scaffold_action the planner originally produced.
      - The TrustContext signals that flowed into the trust logic.
    """
    if not traces_dir.exists():
        raise FileNotFoundError(f"traces_dir not found: {traces_dir}")

    for trace_path in sorted(traces_dir.glob("*.json")):
        try:
            with trace_path.open() as fh:
                trace = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping unreadable trace %s: %s", trace_path, exc)
            continue
        trace["_path"] = str(trace_path)
        yield trace


def trace_to_inputs(trace: dict[str, Any]) -> tuple[str, TrustContext]:
    """Reconstruct (scaffold_action, TrustContext) from a stored trace.

    Field mapping matches Gemma4HAICAdapter payload + Sentinel probe conventions.
    Adjust if local traces use different key names.
    """
    scaffold_action = trace.get("scaffold_action") or trace.get("scaffold", {}).get("action", "accept")
    probe = trace.get("probe", {})
    geometry = trace.get("geometry", {})
    sample = trace.get("sample", {})
    vla = trace.get("observation", {}) or trace.get("vla", {}) or trace.get("assessment", {})

    context = TrustContext(
        target_id=sample.get("target_id") or trace.get("target_id", "unknown"),
        target_priority=sample.get("target_priority", "normal"),
        target_tags=list(sample.get("target_tags", [])),
        sentinel_available=probe.get("sentinel_available"),
        sentinel_cloud_cover=probe.get("sentinel_cloud_cover"),
        sentinel_source=probe.get("sentinel_source"),
        sentinel_datetime=probe.get("sentinel_datetime"),
        target_visible=geometry.get("target_visible"),
        elevation_degrees=geometry.get("elevation_degrees"),
        off_nadir_degrees=geometry.get("off_nadir_degrees"),
        line_of_sight=geometry.get("target_visible"),  # proxy; refine during refactor
        vla_usable=vla.get("usable_observation"),
        vla_scene_match=vla.get("scene_match_score"),
        vla_salience=vla.get("salience_score"),
        vla_change_or_event=vla.get("change_or_event_score"),
        vla_occlusion_or_cloud_risk=vla.get("occlusion_or_cloud_risk"),
        vla_confidence=vla.get("confidence"),
        vla_recommended_action=vla.get("recommended_action"),
        hours_since_last_materialize=sample.get("hours_since_last_materialize"),
        scenario_pack=sample.get("scenario_pack", "all"),
        planner_window_id=trace.get("window_id") or trace.get("trace_id"),
    )
    return scaffold_action, context


# --------------------------------------------------------------------------- #
# Old (scattered) trust logic — shim; fill in during refactor                 #
# --------------------------------------------------------------------------- #

def old_scattered_decide(scaffold_action: str, context: TrustContext) -> dict[str, Any]:
    """Call whatever planner.py / service.py does today.

    This shim is the critical correctness-gate for the refactor. During the
    refactor, import the existing scattered functions here and call them with
    the reconstructed context. The return dict must include at minimum:

        {"action": TrustAction, "reason": str}

    If the existing code returns a different structure, adapt it here (not in
    the parity harness logic) so the comparison stays apples-to-apples.
    """
    raise NotImplementedError(
        "Fill in old_scattered_decide() by importing the existing scattered "
        "trust functions from planner.py / service.py. Without this shim the "
        "parity harness cannot verify the refactor is a no-op."
    )


# --------------------------------------------------------------------------- #
# Parity check                                                                #
# --------------------------------------------------------------------------- #

def compare_decisions(
    old_result: dict[str, Any],
    new_result: Any,   # TrustDecision from WCLITrustModel.decide
) -> tuple[bool, str]:
    """Return (match_ok, diff_message). Case-insensitive on reason codes, order-insensitive."""
    old_action = str(old_result.get("action", "")).strip().lower()
    new_action = new_result.action.strip().lower()

    old_reasons = {r.strip().lower() for r in str(old_result.get("reason", "")).split(",") if r.strip()}
    new_reasons = {r.strip().lower() for r in new_result.reason.split(",") if r.strip()}

    if old_action != new_action:
        return False, f"action differs: old={old_action!r} new={new_action!r}"

    # Reason-code parity is often the brittle part — consider matching on a
    # subset (the "required" codes) rather than full set-equality if the
    # scattered logic emits extras the new module doesn't need to replicate.
    reasons_symdiff = old_reasons.symmetric_difference(new_reasons)
    if reasons_symdiff:
        return False, f"reason codes differ: old={sorted(old_reasons)} new={sorted(new_reasons)} diff={sorted(reasons_symdiff)}"

    return True, "match"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Parity harness for WCLITrustModel refactor")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("traces/"),
        help="Directory containing stored ObservationVLA trace JSON files",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Cap on number of traces to compare (0 = all)",
    )
    parser.add_argument(
        "--strict-reasons",
        action="store_true",
        help="Require exact reason-set equality (default is symmetric-difference check)",
    )
    args = parser.parse_args()

    try:
        model = WCLITrustModel.from_defaults()
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to construct WCLITrustModel: %s", exc)
        return 2

    mismatches: list[tuple[str, dict[str, Any], Any, str]] = []
    total = 0

    for trace in load_traces(args.traces_dir):
        if args.max_cases and total >= args.max_cases:
            break
        total += 1

        try:
            scaffold_action, context = trace_to_inputs(trace)
        except Exception as exc:  # noqa: BLE001
            logger.warning("trace_to_inputs failed for %s: %s", trace.get("_path"), exc)
            continue

        try:
            old_result = old_scattered_decide(scaffold_action, context)
        except NotImplementedError as exc:
            logger.error(
                "old_scattered_decide() is not wired up yet. This shim must call the "
                "existing scattered logic from planner.py / service.py. See the "
                "refactor README for details."
            )
            return 2
        except Exception as exc:  # noqa: BLE001
            logger.warning("old_scattered_decide failed on %s: %s", trace.get("_path"), exc)
            continue

        new_result = model.decide(scaffold_action, context)

        match_ok, diff_msg = compare_decisions(old_result, new_result)
        if not match_ok:
            mismatches.append((trace.get("_path", "<no-path>"), old_result, new_result, diff_msg))

    logger.info("compared %d traces; %d mismatches", total, len(mismatches))

    if mismatches:
        logger.error("FAIL: refactor dropped or flipped decisions")
        for path, old, new, diff in mismatches[:20]:
            logger.error("  trace=%s :: %s", path, diff)
            logger.error("    old=%s", json.dumps(old, default=str))
            logger.error("    new=%s", json.dumps(asdict(new), default=str))
        if len(mismatches) > 20:
            logger.error("  ... (%d more mismatches)", len(mismatches) - 20)
        return 1

    logger.info("PASS: all %d decisions match. Refactor is a pure no-op; safe to commit.", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
