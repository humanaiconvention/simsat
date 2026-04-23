from __future__ import annotations

from observation_vla.schemas import (
    ObservationAssessment,
    ObservationAssessmentRecord,
    ObservationEvidence,
    ObservationMemoryState,
    ObservationOutcome,
    ObservationResidual,
    ObservationSample,
    ObservationTraceRecord,
    SubmissionCasePin,
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


def test_observation_store_round_trip_across_all_payloads(tmp_path):
    store = ObservationStore(tmp_path)
    record = _assessment_record()
    trace = _trace()
    outcome = ObservationOutcome(
        outcome_id="out-1",
        trace_id=trace.trace_id,
        assessment_id=record.assessment.assessment_id,
        decision_id=trace.decision_id,
        registered_at="2026-03-10T12:02:00Z",
        mission_id="mission-1",
        operator_action="approve",
        useful=True,
        usefulness_score=0.91,
        label_source="operator_review",
        reviewer="ben",
        review_status="reviewed",
        outcome_tags=["high_signal"],
    )
    state = ObservationMemoryState(
        state_id="mission:mission-1",
        scope="mission",
        scope_id="mission-1",
        last_updated="2026-03-10T12:03:00Z",
        sample_count=1,
        useful_count=1,
        mean_usefulness=0.91,
        confidence_bias=0.1,
        refine_bias=-0.05,
        action_alignment=1.0,
    )
    pin = SubmissionCasePin(
        scenario_pack="scope-1",
        trace_id=trace.trace_id,
        decision_id=trace.decision_id,
        target_id=trace.target_id,
        target_label=trace.sample.target_label,
        outcome_id=outcome.outcome_id,
        label_source="operator_review",
        reviewer="ben",
        pinned_at="2026-03-10T12:04:00Z",
        pinned_by="ben",
        notes="Best reviewed case",
    )

    store.save_record(record)
    store.save_trace(trace)
    store.save_outcome(outcome)
    store.save_memory_state(state)
    store.save_submission_case(pin)

    assert store.get_record("asm-1") is not None
    assert store.get_trace(trace.trace_id) is not None
    assert store.get_outcome(outcome.outcome_id) is not None
    assert store.get_outcome_for_trace(trace.trace_id) is not None
    assert store.get_memory_state("mission", "mission-1") is not None
    assert store.get_submission_case("scope-1") is not None
    assert store.list_records(limit=5)[0].assessment.assessment_id == "asm-1"
    assert store.list_traces(limit=5)[0].trace_id == trace.trace_id
    assert store.list_outcomes(limit=5)[0].outcome_id == outcome.outcome_id
    assert store.list_memory_states(scope="mission", limit=5)[0].state_id == state.state_id
    assert store.list_submission_cases()[0].trace_id == trace.trace_id


def test_observation_store_current_outcome_and_memory_replacement(tmp_path):
    store = ObservationStore(tmp_path)
    trace = _trace()
    store.save_trace(trace)
    older = ObservationOutcome(
        outcome_id="out-older",
        trace_id=trace.trace_id,
        assessment_id="asm-1",
        decision_id=trace.decision_id,
        registered_at="2026-03-10T12:02:00Z",
        operator_action="refine",
        useful=False,
        usefulness_score=0.2,
        is_current=False,
    )
    current = ObservationOutcome(
        outcome_id="out-current",
        trace_id=trace.trace_id,
        assessment_id="asm-1",
        decision_id=trace.decision_id,
        registered_at="2026-03-10T12:03:00Z",
        operator_action="approve",
        useful=True,
        usefulness_score=0.85,
        is_current=True,
    )

    store.save_outcome(older)
    store.save_outcome(current)
    store.replace_memory_states(
        [
            ObservationMemoryState(
                state_id="scenario:scope-1",
                scope="scenario",
                scope_id="scope-1",
                last_updated="2026-03-10T12:04:00Z",
                sample_count=2,
                useful_count=1,
                mean_usefulness=0.5,
            )
        ]
    )

    assert store.get_outcome_for_trace(trace.trace_id).outcome_id == "out-current"
    assert len(store.list_outcomes(limit=10, current_only=True)) == 1
    assert len(store.list_outcomes(limit=10, current_only=False)) == 2
    assert store.list_memory_states(scope="scenario", limit=5)[0].scope_id == "scope-1"
