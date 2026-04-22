from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .planner import MissionResponsePlanner
from .policy import MissionResponsePolicy
from .schemas import MissionResponseAction, MissionResponseOutcome, MissionUtilitySnapshot
from .store import MissionResponseStore
from .utility import MissionUtilityModel

if TYPE_CHECKING:
    from observation_vla.service import ObservationVLAService


class MissionResponseService:
    def __init__(
        self,
        planner: MissionResponsePlanner,
        utility_model: MissionUtilityModel,
        store: MissionResponseStore,
        policy: MissionResponsePolicy,
        observation_vla: "ObservationVLAService",
    ) -> None:
        self.planner = planner
        self.utility_model = utility_model
        self.store = store
        self.policy = policy
        self.observation_vla = observation_vla

    def propose_from_trace(
        self,
        trace: dict,
        outcome: dict | None = None,
        target_priority: float = 0.5,
        target_label: str = "",
    ) -> MissionResponseAction:
        decision_after = dict(trace.get("decision_after", {}))
        assessment = dict(trace.get("assessment", {}))
        evidence = dict(assessment.get("evidence", {}))
        probe = dict(trace.get("probe", {}))
        effective_action = (
            decision_after.get("effective_action")
            or assessment.get("recommended_action")
            or decision_after.get("action")
            or "skip"
        )
        confidence = float(evidence.get("confidence", 0.0))
        effective_score = float(
            decision_after.get("effective_combined_score")
            if decision_after.get("effective_combined_score") is not None
            else decision_after.get("combined_score", 0.0)
        )
        useful = outcome.get("useful") if outcome is not None else None
        cloud_cover = probe.get("sentinel_cloud_cover")
        utility_score = self.utility_model.score(
            target_priority=target_priority,
            effective_combined_score=effective_score,
            confidence=confidence,
            useful=useful,
            cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
        )
        rationale_tags = list(dict.fromkeys([
            *(decision_after.get("observation_rationale_tags", []) or []),
            *(assessment.get("calibration_tags", []) or []),
            f"effective_action:{effective_action}",
        ]))

        action = self.planner.decide(
            decision_id=str(trace.get("decision_id", "")),
            trace_id=str(trace.get("trace_id", "")) or None,
            scenario_pack=str(trace.get("scenario_pack", "all")),
            target_id=str(trace.get("target_id", "")),
            target_label=target_label or str(trace.get("sample", {}).get("target_label", "")),
            effective_action=str(effective_action),
            effective_score=effective_score,
            confidence=confidence,
            useful=bool(useful) if useful is not None else None,
            cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
            utility_score=utility_score,
            rationale_tags=rationale_tags,
            priority=float(trace.get("features", {}).get("target_priority", target_priority)),
        )
        return self.store.save_action(action)

    def propose_from_trace_id(self, trace_id: str) -> MissionResponseAction:
        existing = self.store.get_action_for_trace(trace_id)
        if existing is not None:
            return existing
        trace, outcome = self.observation_vla.get_trace_with_outcome(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id} not found")
        return self.propose_from_trace(
            trace=trace.to_dict(),
            outcome=outcome.to_dict() if outcome is not None else None,
            target_priority=float(trace.features.get("target_priority", 0.5)),
            target_label=trace.sample.target_label,
        )

    def register_action_outcome(
        self,
        action_id: str,
        utility_realized: float,
        execution_status: str = "simulated",
        notes: str | None = None,
    ) -> MissionResponseOutcome:
        action = self.store.get_action(action_id)
        if action is None:
            raise KeyError(f"Action {action_id} not found")
        executed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        outcome = MissionResponseOutcome(
            action_id=action_id,
            executed_at=executed_at,
            execution_status=execution_status,
            utility_realized=utility_realized,
            notes=notes,
        )
        action.status = "executed"
        self.store.save_action(action)
        return self.store.save_outcome(outcome)

    def summarize_mission(
        self,
        mission_id: str = "default",
        scenario_pack: str = "all",
    ) -> MissionUtilitySnapshot:
        actions = [
            action for action in self.store.list_actions(limit=500)
            if action.scenario_pack == scenario_pack
        ]
        action_counts: dict[str, int] = {}
        utility_realized = 0.0
        utility_missed = 0.0
        outcomes_by_action = {outcome.action_id: outcome for outcome in self.store.list_outcomes(limit=500)}
        for action in actions:
            action_counts[action.recommended_action] = action_counts.get(action.recommended_action, 0) + 1
            outcome = outcomes_by_action.get(action.action_id)
            if outcome is None:
                utility_missed += action.utility_score
            else:
                utility_realized += outcome.utility_realized
                utility_missed += max(action.utility_score - outcome.utility_realized, 0.0)
        snapshot = MissionUtilitySnapshot(
            mission_id=mission_id,
            scenario_pack=scenario_pack,
            total_actions=len(actions),
            utility_realized=utility_realized,
            utility_missed=utility_missed,
            action_counts=action_counts,
        )
        return self.store.save_snapshot(snapshot)

    def list_actions(self, limit: int = 50) -> list[MissionResponseAction]:
        return self.store.list_actions(limit=limit)

    def list_outcomes(self, limit: int = 50) -> list[MissionResponseOutcome]:
        return self.store.list_outcomes(limit=limit)

    def get_trace_response(self, trace_id: str) -> dict[str, Any]:
        action = self.store.get_action_for_trace(trace_id)
        if action is None:
            raise KeyError(f"No mission response action found for trace {trace_id}")
        outcome = self.store.get_outcome_for_action(action.action_id)
        return {
            "action": action.to_dict(),
            "outcome": outcome.to_dict() if outcome is not None else None,
        }
