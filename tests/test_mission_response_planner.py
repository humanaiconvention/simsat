from __future__ import annotations

from mission_response.planner import MissionResponsePlanner
from mission_response.policy import MissionResponsePolicy


def test_decide_escalates_operator_for_high_utility_low_confidence_trace():
    planner = MissionResponsePlanner(MissionResponsePolicy())

    action = planner.decide(
        decision_id="dec-1",
        trace_id="trace-1",
        scenario_pack="scope-1",
        target_id="target-1",
        target_label="Target One",
        effective_action="accept",
        effective_score=0.9,
        confidence=0.4,
        useful=True,
        cloud_cover=10.0,
        utility_score=0.75,
        rationale_tags=["base_tag"],
        priority=0.9,
    )

    assert action.recommended_action == "escalate_operator"
    assert "operator_escalation" in action.rationale_tags
    assert "labelled_useful" in action.rationale_tags
    assert "low_confidence" in action.rationale_tags


def test_decide_materializes_high_utility_accept_case():
    planner = MissionResponsePlanner(MissionResponsePolicy())

    action = planner.decide(
        decision_id="dec-2",
        trace_id=None,
        scenario_pack="scope-2",
        target_id="target-2",
        target_label="Target Two",
        effective_action="accept",
        effective_score=0.88,
        confidence=0.82,
        useful=None,
        cloud_cover=5.0,
        utility_score=0.9,
        rationale_tags=["base_tag"],
        priority=0.8,
    )

    assert action.recommended_action == "materialize_now"
    assert "high_utility_accept" in action.rationale_tags
    assert action.created_at.endswith("Z")
