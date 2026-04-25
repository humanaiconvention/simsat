from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "src" / "sim" / "data" / "observation_vla" / "traces.json"
OUTCOME_PATH = ROOT / "src" / "sim" / "data" / "observation_vla" / "outcomes.json"


def _load_traces() -> dict[str, dict]:
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    return {trace["trace_id"]: trace for trace in payload["traces"]}


def _load_outcomes() -> list[dict]:
    payload = json.loads(OUTCOME_PATH.read_text(encoding="utf-8"))
    return payload["outcomes"]


def test_backlog_shortlist_trace_fixtures_exist_with_expected_runtime_modes():
    traces = _load_traces()

    houston = traces["trace_cbe073faf8554afb986b632128204650"]
    singapore = traces["trace_e9e45ad652ff4e74853f0c6d0c03a827"]
    sf_bay = traces["trace_dbbe33015d04413082df17ba2f0c8d12"]

    assert houston["scenario_pack"] == "disaster_response_weather"
    assert houston["assessment"]["runtime_mode"] == "clip_local"
    assert houston["assessment"]["recommended_action"] == "refine"

    assert singapore["scenario_pack"] == "maritime_chokepoints"
    assert singapore["target_id"] == "singapore_port"
    assert singapore["assessment"]["runtime_mode"] == "clip_local"
    assert singapore["decision_after"]["effective_action"] == "refine"

    assert sf_bay["scenario_pack"] == "urban_coastal_ambiguity"
    assert sf_bay["assessment"]["recommended_action"] == "accept"
    assert sf_bay["decision_after"]["effective_action"] == "refine"


def test_local_submission_case_trace_and_outcomes_have_expected_shape():
    traces = _load_traces()
    outcomes = _load_outcomes()

    trace = traces["trace_e5a00b774bcf410fafe9944b8a4cc4f8"]
    outcome = next(item for item in outcomes if item["trace_id"] == "trace_e5a00b774bcf410fafe9944b8a4cc4f8")

    assert trace["assessment"]["runtime_mode"] == "stub"
    assert trace["assessment"]["assessment_mode"] == "image_conditioned"
    assert trace["assessment"]["recommended_action"] == "accept"
    assert trace["residual"]["effective_action"] == "accept"
    assert trace["sample"]["images"][0]["image_type"] == "sentinel_rgb"
    assert outcome["operator_action"] == "accept"
    assert outcome["useful"] is True
    assert outcome["usefulness_score"] == 0.92


