from __future__ import annotations

import pytest
from fastapi import HTTPException

from observation_vla import api_router as observation_api


class FakeObservationVLAService:
    def list_assessments(self, limit=20):
        return [type("AssessmentRecord", (), {"to_dict": lambda self: {"assessment_id": "asm-1"}})()]

    def get_assessment(self, assessment_id):
        if assessment_id == "missing":
            return None
        return type("AssessmentRecord", (), {"to_dict": lambda self: {"assessment_id": assessment_id}})()

    def list_traces(self, limit=20):
        return [type("TraceRecord", (), {"to_dict": lambda self: {"trace_id": "trace-1"}})()]

    def get_trace_with_outcome(self, trace_id):
        if trace_id == "missing":
            return None, None
        trace = type("TraceRecord", (), {"to_dict": lambda self: {"trace_id": trace_id}})()
        outcome = type("Outcome", (), {"to_dict": lambda self: {"outcome_id": "out-1"}})()
        return trace, outcome

    def reassess_trace(self, trace_id, persist=False):
        if trace_id == "missing":
            raise KeyError("Trace missing")
        return type("AssessmentRecord", (), {"to_dict": lambda self: {"assessment_id": "asm-1", "persist": persist}})()

    def register_outcome(self, **kwargs):
        outcome = type("Outcome", (), {"to_dict": lambda self: {"trace_id": kwargs["trace_id"], "label_source": kwargs["label_source"]}})()
        state = type("State", (), {"to_dict": lambda self: {"state_id": "mission:default"}})()
        trace = type("Trace", (), {"to_dict": lambda self: {"trace_id": kwargs["trace_id"]}})()
        return outcome, [state], trace

    def register_operator_review(self, **kwargs):
        outcome = type("Outcome", (), {"to_dict": lambda self: {"trace_id": kwargs["trace_id"], "reviewer": kwargs["reviewer"]}})()
        state = type("State", (), {"to_dict": lambda self: {"state_id": "mission:default"}})()
        trace = type("Trace", (), {"to_dict": lambda self: {"trace_id": kwargs["trace_id"]}})()
        return outcome, [state], trace

    def list_outcomes(self, limit=20, current_only=True):
        return [type("Outcome", (), {"to_dict": lambda self: {"outcome_id": "out-1"}})()]

    def list_review_candidates(self, scenario_pack=None, limit=20):
        return [{"trace_id": "trace-1", "scenario_pack": scenario_pack or "all"}]

    def get_review_bundle(self, trace_id):
        if trace_id == "missing":
            raise KeyError("Trace missing")
        return {"trace": {"trace_id": trace_id}}

    def list_submission_cases(self):
        return [type("Case", (), {"to_dict": lambda self: {"scenario_pack": "scope-1"}})()]

    def pin_submission_case(self, scenario_pack, trace_id, pinned_by=None, notes=None):
        if trace_id == "missing":
            raise KeyError("Trace missing")
        return type("Case", (), {"to_dict": lambda self: {"scenario_pack": scenario_pack, "trace_id": trace_id}})()

    def list_memory_states(self, scope=None, limit=20):
        return [type("State", (), {"to_dict": lambda self: {"scope": scope or "mission"}})()]

    def export_dataset(self, limit=100, labelled_only=True):
        return [{"trace_id": "trace-1", "labelled_only": labelled_only}]


@pytest.mark.anyio
async def test_observation_vla_router_requires_initialization(monkeypatch):
    monkeypatch.setattr(observation_api, "_service", None)
    with pytest.raises(HTTPException) as exc:
        await observation_api.list_assessments()
    assert exc.value.status_code == 503


@pytest.mark.anyio
async def test_observation_vla_router_happy_path(monkeypatch):
    monkeypatch.setattr(observation_api, "_service", FakeObservationVLAService())

    assessments = await observation_api.list_assessments()
    trace = await observation_api.get_trace("trace-1")
    outcome = await observation_api.register_outcome(
        "trace-1",
        observation_api.OutcomeRequest(
            operator_action="approve",
            useful=True,
            usefulness_score=0.9,
            label_source="simulated",
        ),
    )
    review = await observation_api.register_operator_review(
        "trace-1",
        observation_api.OperatorReviewRequest(
            reviewer="ben",
            operator_action="approve",
            useful=True,
            usefulness_score=0.95,
        ),
    )
    dataset = await observation_api.export_dataset(labelled_only=True)

    assert assessments["assessments"][0]["assessment_id"] == "asm-1"
    assert trace["current_outcome"]["outcome_id"] == "out-1"
    assert outcome["outcome"]["trace_id"] == "trace-1"
    assert review["outcome"]["reviewer"] == "ben"
    assert dataset["examples"][0]["labelled_only"] is True


@pytest.mark.anyio
async def test_observation_vla_router_error_translation(monkeypatch):
    monkeypatch.setattr(observation_api, "_service", FakeObservationVLAService())

    with pytest.raises(HTTPException) as exc:
        await observation_api.get_assessment("missing")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc2:
        await observation_api.get_review_bundle("missing")
    assert exc2.value.status_code == 404
