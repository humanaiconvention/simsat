from __future__ import annotations

from typing import Any

from .schemas import ObservationSample

TRIAGE_PROMPT_VERSION = "obs-vla-triage-v1"


def expected_response_schema() -> dict[str, Any]:
    return {
        "usable_observation": "bool",
        "scene_match_score": "float_0_1",
        "salience_score": "float_0_1",
        "change_or_event_score": "float_0_1",
        "occlusion_or_cloud_risk": "float_0_1",
        "confidence": "float_0_1",
        "recommended_action": "accept|defer|refine|skip",
        "rationale_tags": ["string"],
    }


def build_triage_prompt(sample: ObservationSample) -> str:
    metadata_note = sample.target_metadata.get("challenge_note") if isinstance(sample.target_metadata, dict) else None
    scenario_role = sample.target_metadata.get("scenario_role") if isinstance(sample.target_metadata, dict) else None
    return (
        "Assess this Earth observation for mission usefulness.\n"
        f"Scenario pack: {sample.scenario_pack}\n"
        f"Target: {sample.target_label or sample.target_id}\n"
        f"Target tags: {sample.target_tags}\n"
        f"Scenario role: {scenario_role}\n"
        f"Challenge note: {metadata_note}\n"
        f"Geometry: {sample.geometry}\n"
        f"Probe: {sample.probe}\n"
        "Return a compact structured assessment matching the requested schema."
    )
