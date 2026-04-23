from __future__ import annotations

from types import SimpleNamespace

from mission_response.planner import MissionResponsePlanner
from mission_response.policy import MissionResponsePolicy
from mission_response.service import MissionResponseService
from mission_response.store import MissionResponseStore
from mission_response.utility import MissionUtilityModel


class FakeSerializable:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class FakeObservationVLA:
    def __init__(self) -> None:
        self.trace_calls: list[str] = []
        self.trace = SimpleNamespace(
            to_dict=lambda: {
                "decision_id": "dec-trace",
                "trace_id": "trace-1",
                "scenario_pack": "scope-1",
                "target_id": "target-1",
                "sample": {"target_label": "Trace Target"},
                "features": {"target_priority": 0.92},
                "decision_after": {
                    "effective_action": "accept",
                    "effective_combined_score": 0.88,
                    "observation_rationale_tags": ["obs_signal"],
                },
                "assessment": {
                    "recommended_action": "materialize_now",
                    "evidence": {"confidence": 0.81},
                    "calibration_tags": ["calibrated"],
                },
                "probe": {"sentinel_cloud_cover": 4.0},
            },
            features={"target_priority": 0.92},
            sample=SimpleNamespace(target_label="Trace Target"),
        )
        self.outcome = FakeSerializable(
            {
                "trace_id": "trace-1",
                "useful": True,
                "usefulness_score": 0.87,
                "operator_action": "approve",
            }
        )

    def get_trace_with_outcome(self, trace_id: str):
        self.trace_calls.append(trace_id)
        if trace_id != "trace-1":
            return None, None
        return self.trace, self.outcome


def _service(tmp_path, observation_vla: FakeObservationVLA | None = None) -> MissionResponseService:
    policy = MissionResponsePolicy()
    return MissionResponseService(
        planner=MissionResponsePlanner(policy),
        utility_model=MissionUtilityModel(),
        store=MissionResponseStore(tmp_path),
        policy=policy,
        observation_vla=observation_vla or FakeObservationVLA(),
    )


def test_propose_from_trace_persists_action_and_tags(tmp_path):
    service = _service(tmp_path)

    action = service.propose_from_trace(
        trace={
            "decision_id": "dec-1",
            "trace_id": "trace-1",
            "scenario_pack": "scope-1",
            "target_id": "target-1",
            "sample": {"target_label": "Manual Target"},
            "features": {"target_priority": 0.95},
            "decision_after": {
                "effective_action": "accept",
                "effective_combined_score": 0.91,
                "observation_rationale_tags": ["obs_signal"],
            },
            "assessment": {
                "recommended_action": "materialize_now",
                "evidence": {"confidence": 0.83},
                "calibration_tags": ["calibrated"],
            },
            "probe": {"sentinel_cloud_cover": 3.0},
        },
        outcome={"useful": True},
        target_priority=0.95,
        target_label="Manual Target",
    )

    persisted = service.store.get_action(action.action_id)
    assert persisted is not None
    assert action.recommended_action == "materialize_now"
    assert "obs_signal" in action.rationale_tags
    assert "calibrated" in action.rationale_tags
    assert "effective_action:accept" in action.rationale_tags
    assert action.target_label == "Manual Target"


def test_propose_from_trace_id_uses_existing_action_before_observation_lookup(tmp_path):
    observation_vla = FakeObservationVLA()
    service = _service(tmp_path, observation_vla=observation_vla)
    existing = service.propose_from_trace(
        trace=observation_vla.trace.to_dict(),
        outcome=observation_vla.outcome.to_dict(),
        target_priority=0.92,
        target_label="Trace Target",
    )

    loaded = service.propose_from_trace_id("trace-1")

    assert loaded.action_id == existing.action_id
    assert observation_vla.trace_calls == []


def test_register_action_outcome_and_summarize_mission(tmp_path):
    service = _service(tmp_path)
    first = service.propose_from_trace(
        trace={
            "decision_id": "dec-1",
            "trace_id": "trace-1",
            "scenario_pack": "scope-1",
            "target_id": "target-1",
            "sample": {"target_label": "Target One"},
            "features": {"target_priority": 0.9},
            "decision_after": {"effective_action": "accept", "effective_combined_score": 0.88},
            "assessment": {"evidence": {"confidence": 0.8}, "calibration_tags": []},
            "probe": {"sentinel_cloud_cover": 5.0},
        },
        outcome={"useful": True},
        target_priority=0.9,
        target_label="Target One",
    )
    second = service.propose_from_trace(
        trace={
            "decision_id": "dec-2",
            "trace_id": "trace-2",
            "scenario_pack": "scope-1",
            "target_id": "target-2",
            "sample": {"target_label": "Target Two"},
            "features": {"target_priority": 0.6},
            "decision_after": {"effective_action": "refine", "effective_combined_score": 0.55},
            "assessment": {"evidence": {"confidence": 0.45}, "calibration_tags": []},
            "probe": {"sentinel_cloud_cover": 50.0},
        },
        outcome=None,
        target_priority=0.6,
        target_label="Target Two",
    )

    outcome = service.register_action_outcome(first.action_id, utility_realized=0.7, execution_status="simulated")
    summary = service.summarize_mission(mission_id="mission-1", scenario_pack="scope-1")
    trace_response = service.get_trace_response("trace-1")

    assert outcome.action_id == first.action_id
    assert service.store.get_action(first.action_id).status == "executed"
    assert summary.total_actions == 2
    assert summary.utility_realized == 0.7
    assert summary.utility_missed > 0.0
    assert summary.action_counts[first.recommended_action] == 1
    assert summary.action_counts[second.recommended_action] == 1
    assert trace_response["action"]["action_id"] == first.action_id
    assert trace_response["outcome"]["action_id"] == first.action_id
