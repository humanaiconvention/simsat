from __future__ import annotations

from datetime import datetime, timezone

from .policy import MissionResponsePolicy
from .schemas import MissionResponseAction


class MissionResponsePlanner:
    def __init__(self, policy: MissionResponsePolicy) -> None:
        self.policy = policy

    def decide(
        self,
        *,
        decision_id: str,
        trace_id: str | None,
        scenario_pack: str,
        target_id: str,
        target_label: str,
        effective_action: str,
        effective_score: float,
        confidence: float,
        useful: bool | None,
        cloud_cover: float | None,
        utility_score: float,
        rationale_tags: list[str],
        priority: float,
    ) -> MissionResponseAction:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        recommended_action = "drop"
        tags = list(rationale_tags)

        if (
            effective_action in {"accept", "refine"}
            and utility_score >= self.policy.operator_escalation_threshold
            and confidence < 0.55
        ):
            recommended_action = "escalate_operator"
            tags.append("operator_escalation")
        elif effective_action == "accept" and utility_score >= self.policy.materialize_now_threshold:
            recommended_action = "materialize_now"
            tags.append("high_utility_accept")
        elif effective_action in {"refine", "defer"} and utility_score >= self.policy.refine_review_threshold:
            recommended_action = "queue_refine_review"
            tags.append("review_queue")
        elif utility_score >= self.policy.revisit_threshold:
            recommended_action = "schedule_revisit"
            tags.append("revisit_candidate")
        else:
            recommended_action = "drop"
            tags.append("low_utility_drop")

        if useful is False:
            tags.append("labelled_not_useful")
        elif useful is True:
            tags.append("labelled_useful")
        if cloud_cover is not None and cloud_cover > 40.0:
            tags.append("cloud_risk")
        if confidence < 0.5:
            tags.append("low_confidence")

        return MissionResponseAction(
            decision_id=decision_id,
            trace_id=trace_id,
            scenario_pack=scenario_pack,
            target_id=target_id,
            target_label=target_label,
            recommended_action=recommended_action,
            priority=priority,
            utility_score=utility_score,
            confidence=confidence,
            rationale_tags=list(dict.fromkeys(tags)),
            status="proposed",
            created_at=created_at,
        )
