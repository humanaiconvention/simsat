from __future__ import annotations

from mission_response.schemas import MissionResponseAction, MissionResponseOutcome, MissionUtilitySnapshot
from mission_response.store import MissionResponseStore


def test_mission_response_store_round_trip(tmp_path):
    store = MissionResponseStore(tmp_path)

    action = MissionResponseAction(
        action_id="resp-1",
        decision_id="dec-1",
        trace_id="trace-1",
        scenario_pack="scope-1",
        target_id="target-1",
        target_label="Target One",
        recommended_action="materialize_now",
        priority=0.9,
        utility_score=0.84,
        confidence=0.78,
        rationale_tags=["high_utility_accept"],
        created_at="2026-03-10T12:00:00Z",
    )
    outcome = MissionResponseOutcome(
        outcome_id="out-1",
        action_id="resp-1",
        executed_at="2026-03-10T12:05:00Z",
        execution_status="simulated",
        utility_realized=0.8,
        notes="Successful follow-through",
    )
    snapshot = MissionUtilitySnapshot(
        mission_id="mission-1",
        scenario_pack="scope-1",
        total_actions=1,
        utility_realized=0.8,
        utility_missed=0.1,
        action_counts={"materialize_now": 1},
    )

    store.save_action(action)
    store.save_outcome(outcome)
    store.save_snapshot(snapshot)

    assert store.get_action("resp-1") is not None
    assert store.get_action_for_trace("trace-1") is not None
    assert store.get_outcome_for_action("resp-1") is not None
    assert store.get_snapshot("mission-1", "scope-1") is not None

    listed_actions = store.list_actions(limit=5)
    listed_outcomes = store.list_outcomes(limit=5)

    assert listed_actions[0].recommended_action == "materialize_now"
    assert listed_outcomes[0].utility_realized == 0.8


def test_mission_response_store_keeps_latest_action_for_trace(tmp_path):
    store = MissionResponseStore(tmp_path)

    older = MissionResponseAction(
        action_id="resp-older",
        decision_id="dec-1",
        trace_id="trace-1",
        scenario_pack="scope-1",
        target_id="target-1",
        target_label="Target One",
        recommended_action="schedule_revisit",
        created_at="2026-03-10T12:00:00Z",
    )
    newer = MissionResponseAction(
        action_id="resp-newer",
        decision_id="dec-2",
        trace_id="trace-1",
        scenario_pack="scope-1",
        target_id="target-1",
        target_label="Target One",
        recommended_action="escalate_operator",
        created_at="2026-03-10T12:10:00Z",
    )

    store.save_action(older)
    store.save_action(newer)

    latest = store.get_action_for_trace("trace-1")
    assert latest is not None
    assert latest.action_id == "resp-newer"
