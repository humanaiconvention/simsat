from __future__ import annotations

import pytest
from fastapi import HTTPException

from encounter import api_router as encounter_api
from mission_response import api_router as mission_api


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


@pytest.mark.anyio
async def test_encounter_router_requires_initialization(monkeypatch):
    monkeypatch.setattr(encounter_api, "_service", None)
    with pytest.raises(HTTPException) as exc:
        await encounter_api.get_policy()
    assert exc.value.status_code == 503


@pytest.mark.anyio
async def test_encounter_router_happy_path(monkeypatch):
    monkeypatch.setattr(encounter_api, "_service", FakeEncounterService())

    policy = await encounter_api.get_policy()
    windows = await encounter_api.get_windows(scenario_pack="scope-1")
    planned = await encounter_api.plan_encounters(encounter_api.PlanRequest(scenario_pack="scope-1"))

    assert policy["policy_id"] == "encounter-test"
    assert windows["total" if "total" in windows else "targets"] == 1
    assert planned["decisions"][0]["decision_id"] == "dec-1"


@pytest.mark.anyio
async def test_mission_response_router_happy_path(monkeypatch):
    monkeypatch.setattr(mission_api, "_service", FakeMissionResponseService())

    actions = await mission_api.list_actions()
    proposed = await mission_api.propose_from_trace("trace-1")
    registered = await mission_api.register_action_outcome(
        "act-1",
        mission_api.ActionOutcomeRequest(utility_realized=0.7, execution_status="simulated", notes="ok"),
    )
    summary = await mission_api.summary(mission_id="mission-1", scenario_pack="scope-1")

    assert actions["actions"][0]["action_id"] == "act-1"
    assert proposed["action_id"] == "act-trace-1"
    assert registered["action_id"] == "act-1"
    assert summary["mission_id"] == "mission-1"
