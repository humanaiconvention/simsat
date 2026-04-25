from __future__ import annotations

import json

from encounter.schemas import EncounterEvaluation, EncounterEvaluationPrimitive
from encounter.store import EncounterStore
from observation_vla.schemas import (
    ObservationAssessment,
    ObservationAssessmentRecord,
    ObservationEvidence,
    ObservationResidual,
    ObservationSample,
    ObservationTraceRecord,
)
from observation_vla.store import ObservationStore


def _assessment_record() -> ObservationAssessmentRecord:
    return ObservationAssessmentRecord(
        sample=ObservationSample(
            sample_id="obs-1",
            decision_id="dec-1",
            window_id="window-1",
            target_id="target-1",
            scenario_pack="scope-1",
            target_label="Target One",
        ),
        assessment=ObservationAssessment(
            assessment_id="asm-1",
            sample_id="obs-1",
            model_id="fake-vlm",
            created_at="2026-03-10T12:00:00Z",
            assessment_mode="image_conditioned",
            runtime_mode="test_runtime",
            image_count=1,
            recommended_action="accept",
            evidence=ObservationEvidence(confidence=0.9, rationale_tags=["image_backed"]),
        ),
    )


def _trace(trace_id: str = "trace-1") -> ObservationTraceRecord:
    return ObservationTraceRecord(
        trace_id=trace_id,
        created_at="2026-03-10T12:01:00Z",
        scenario_pack="scope-1",
        decision_id="dec-1",
        target_id="target-1",
        window_id="window-1",
        sample=_assessment_record().sample,
        assessment=_assessment_record().assessment,
        residual=ObservationResidual(
            assessment_id="asm-1",
            trust_delta=0.1,
            score_delta=0.05,
            recommended_action="accept",
            effective_action="accept",
            rationale_tags=["residual_applied"],
        ),
        decision_before={"action": "defer"},
        decision_after={"action": "accept", "effective_action": "accept"},
    )


def test_encounter_store_persists_primitive_cache_and_exact_evaluation(tmp_path):
    store = EncounterStore(tmp_path / "encounter")
    primitive = EncounterEvaluationPrimitive(
        cache_key=store.build_primitive_cache_key(
            policy_version="policy-v1",
            scenario_pack="scope-1",
            start_time="2026-03-10T12:00:00Z",
            hours=8.0,
            step_seconds=120,
            top_k=8,
        ),
        created_at="2026-03-10T12:00:00Z",
        policy_version="policy-v1",
        scenario_pack="scope-1",
        parameters={"start_time": "2026-03-10T12:00:00Z"},
        window_count=2,
    )
    evaluation = EncounterEvaluation(
        evaluation_id="eval-1",
        created_at="2026-03-10T12:05:00Z",
        parameters={
            "policy_version": "policy-v1",
            "start_time": "2026-03-10T12:00:00Z",
            "hours": 8.0,
            "step_seconds": 120,
            "top_k": 8,
            "materialize_top_k": 2,
        },
        scenario_pack="scope-1",
        window_count=2,
    )

    store.save_evaluation_primitive(primitive)
    store.save_evaluation(
        evaluation,
        full_cache_key=store.build_full_cache_key(
            policy_version="policy-v1",
            scenario_pack="scope-1",
            start_time="2026-03-10T12:00:00Z",
            hours=8.0,
            step_seconds=120,
            top_k=8,
            materialize_top_k=2,
        ),
    )

    assert store.get_evaluation_primitive(primitive.cache_key) is not None
    assert store.get_evaluation_by_cache_key(
        store.build_full_cache_key(
            policy_version="policy-v1",
            scenario_pack="scope-1",
            start_time="2026-03-10T12:00:00Z",
            hours=8.0,
            step_seconds=120,
            top_k=8,
            materialize_top_k=2,
        )
    ).evaluation_id == "eval-1"


def test_observation_store_migrates_legacy_json_into_sqlite_and_filters_by_scenario(tmp_path):
    base = tmp_path / "observation_vla"
    base.mkdir(parents=True, exist_ok=True)
    record = _assessment_record()
    trace = _trace()
    other_trace = _trace("trace-2")
    other_trace.scenario_pack = "scope-2"
    other_trace.sample.scenario_pack = "scope-2"

    (base / "assessments.json").write_text(
        json.dumps({"records": [record.to_dict()]}),
        encoding="utf-8",
    )
    (base / "traces.json").write_text(
        json.dumps({"traces": [trace.to_dict(), other_trace.to_dict()]}),
        encoding="utf-8",
    )
    (base / "outcomes.json").write_text(json.dumps({"outcomes": []}), encoding="utf-8")
    (base / "memory.json").write_text(json.dumps({"states": []}), encoding="utf-8")
    (base / "submission_cases.json").write_text(json.dumps({"cases": []}), encoding="utf-8")

    store = ObservationStore(base)

    assert store.get_record("asm-1") is not None
    filtered = store.list_traces(limit=10, scenario_pack="scope-1")
    assert [item.trace_id for item in filtered] == ["trace-1"]
