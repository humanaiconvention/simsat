"""Re-assess specific stub-runtime pinned traces with the current clip_local adapter.

Targets the 2 remaining stub-runtime entries in SUBMISSION_CASEBOOK.md:
  - Houston Ship Channel  (trace_1926b646ee4b48478913681a33fcdfb1)
  - Suez Canal            (trace_2507337b7939460ebf01cbc9fcef8055)

Loads each trace, fetches its stimulus, re-runs ObservationAssessor.assess() with
the active adapter (clip_local by default), and writes the new assessment back to
the SQLite store via store.save_trace().  The trace's recommended_action and
evidence will reflect the live model — the operator-reviewed outcome (in the
outcomes table) is unaffected, so the casebook narrative still holds, but the
'Observation runtime' field will now read 'clip_local'.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = ROOT / "src" / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from haic.stimulus_store import get_stimulus_store  # type: ignore
from observation_vla.adapter import ObservationVLMAdapter  # type: ignore
from observation_vla.assessor import ObservationAssessor  # type: ignore
from observation_vla.store import ObservationStore  # type: ignore

TARGET_TRACE_IDS = [
    "trace_1926b646ee4b48478913681a33fcdfb1",  # Houston
    "trace_2507337b7939460ebf01cbc9fcef8055",  # Suez
]


def main() -> int:
    store = ObservationStore()
    adapter = ObservationVLMAdapter()
    print(f"Adapter runtime_mode: {adapter.runtime_mode}")
    if adapter.runtime_mode == "stub":
        print("ERROR: adapter is still in stub mode — clip_local model not loaded.", file=sys.stderr)
        return 1

    assessor = ObservationAssessor(adapter, model_id=adapter.runtime_mode)
    stim_store = get_stimulus_store()

    for trace_id in TARGET_TRACE_IDS:
        trace = store.get_trace(trace_id)
        if trace is None:
            print(f"  [skip] {trace_id}: not found in store")
            continue
        print(f"\n=== {trace_id}")
        print(f"  scenario_pack: {trace.scenario_pack}")
        print(f"  target: {trace.sample.target_label}")
        print(f"  prior runtime_mode: {trace.assessment.runtime_mode}")
        print(f"  prior recommended_action: {trace.assessment.recommended_action}")

        stimulus = stim_store.get(trace.stimulus_id) if trace.stimulus_id else None
        if stimulus is None:
            print(f"  [skip] no stimulus available for {trace.stimulus_id}")
            continue

        rendered = assessor.extract_renderable_images(stimulus)
        new_assessment = assessor.assess(trace.sample, rendered_images=rendered)

        # Preserve the original assessment_id so the outcome row (which references it)
        # remains linked.  Also preserve created_at to avoid spurious "newer" sort order.
        new_assessment.assessment_id = trace.assessment.assessment_id
        new_assessment.created_at = trace.assessment.created_at
        # Preserve calibration tags + confidence_adjustment from prior record
        new_assessment.calibration_tags = list(trace.assessment.calibration_tags)
        new_assessment.confidence_adjustment = trace.assessment.confidence_adjustment

        # Replace assessment field on the trace and persist
        trace.assessment = new_assessment
        store.save_trace(trace)
        print(f"  new runtime_mode: {trace.assessment.runtime_mode}")
        print(f"  new recommended_action: {trace.assessment.recommended_action}")
        print(f"  new confidence: {trace.assessment.evidence.confidence:.3f}")
        print(f"  saved.")

    print("\nDone.  Re-run scripts/submission_casebook.py --inprocess to refresh the casebook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
