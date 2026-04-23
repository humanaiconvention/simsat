from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from encounter import api_router as encounter_api
from mission_response import api_router as mission_api
from observation_vla import api_router as observation_api


class FakeEncounterService:
    def get_policy(self):
        return type("Policy", (), {"policy_id": "encounter-test", "to_dict": lambda self: {"policy_id": "encounter-test"}})()

    def list_targets(self):
        return [type("Target", (), {"to_dict": lambda self: {"target_id": "target-1"}})()]

    def list_scenario_packs(self):
        return ["scope-1"]

    def preview_windows(self, **kwargs):
        return [type("Window", (), {"to_dict": lambda self: {"window_id": "window-1"}})()]

    def list_targets_for_scenario(self, scenario_pack=None):
        return [object()]

    def plan(self, **kwargs):
        decision = type("Decision", (), {"to_dict": lambda self: {"decision_id": "dec-1"}})()
        return [type("Record", (), {"decision": decision})()]

    def evaluate(self, **kwargs):
        return type("Evaluation", (), {"to_dict": lambda self: {"evaluation_id": "eval-1"}})()

    def list_evaluations(self, **kwargs):
        return [type("Evaluation", (), {"to_dict": lambda self: {"evaluation_id": "eval-1"}})()]

    def list_decisions(self, **kwargs):
        return [type("Decision", (), {"to_dict": lambda self: {"decision_id": "dec-1"}})()]


class FakeMissionResponseService:
    def list_actions(self, **kwargs):
        return [type("Action", (), {"to_dict": lambda self: {"action_id": "act-1"}})()]

    def list_outcomes(self, **kwargs):
        return [type("Outcome", (), {"to_dict": lambda self: {"outcome_id": "out-1"}})()]

    def propose_from_trace_id(self, trace_id):
        return type("Action", (), {"to_dict": lambda self: {"action_id": f"act-{trace_id}"}})()

    def get_trace_response(self, trace_id):
        return {"action": {"action_id": f"act-{trace_id}"}, "outcome": None}

    def register_action_outcome(self, **kwargs):
        return type("Outcome", (), {"to_dict": lambda self: {"action_id": kwargs["action_id"], "utility_realized": kwargs["utility_realized"]}})()

    def summarize_mission(self, **kwargs):
        return type("Summary", (), {"to_dict": lambda self: {"mission_id": kwargs["mission_id"], "scenario_pack": kwargs["scenario_pack"]}})()


class FakeObservationVLAService:
    def list_assessments(self, limit=20):
        return [type("AssessmentRecord", (), {"to_dict": lambda self: {"assessment_id": "asm-1"}})()]

    def get_assessment(self, assessment_id):
        return type("AssessmentRecord", (), {"to_dict": lambda self: {"assessment_id": assessment_id}})()

    def list_traces(self, limit=20):
        return [type("TraceRecord", (), {"to_dict": lambda self: {"trace_id": "trace-1"}})()]

    def get_trace_with_outcome(self, trace_id):
        trace = type("TraceRecord", (), {"to_dict": lambda self: {"trace_id": trace_id}})()
        outcome = type("Outcome", (), {"to_dict": lambda self: {"outcome_id": "out-1"}})()
        return trace, outcome

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
        return {"trace": {"trace_id": trace_id}}

    def list_submission_cases(self):
        return [type("Case", (), {"to_dict": lambda self: {"scenario_pack": "scope-1"}})()]

    def pin_submission_case(self, scenario_pack, trace_id, pinned_by=None, notes=None):
        return type("Case", (), {"to_dict": lambda self: {"scenario_pack": scenario_pack, "trace_id": trace_id}})()

    def list_memory_states(self, scope=None, limit=20):
        return [type("State", (), {"to_dict": lambda self: {"scope": scope or "mission"}})()]

    def export_dataset(self, limit=100, labelled_only=True):
        return [{"trace_id": "trace-1", "labelled_only": labelled_only}]


def test_testclient_endpoints_return_json_payloads():
    encounter_api.init(FakeEncounterService())
    mission_api.init(FakeMissionResponseService())
    observation_api.init(FakeObservationVLAService())

    app = FastAPI()
    app.include_router(encounter_api.router)
    app.include_router(mission_api.router)
    app.include_router(observation_api.router)
    client = TestClient(app)

    assert client.get("/encounter/policy").json()["policy_id"] == "encounter-test"
    assert client.get("/encounter/windows", params={"scenario_pack": "scope-1"}).json()["windows"][0]["window_id"] == "window-1"
    assert client.post("/mission-response/action/act-1/outcome", json={"utility_realized": 0.7, "execution_status": "simulated", "notes": "ok"}).json()["action_id"] == "act-1"
    assert client.get("/observation-vla/dataset").json()["examples"][0]["labelled_only"] is True
