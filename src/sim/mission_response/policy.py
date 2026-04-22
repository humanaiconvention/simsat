from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MissionResponsePolicy:
    policy_id: str = "mission-response-v1"
    materialize_now_threshold: float = 0.78
    refine_review_threshold: float = 0.58
    revisit_threshold: float = 0.42
    operator_escalation_threshold: float = 0.70
    revisit_bonus: float = 0.08
    operator_bonus: float = 0.10
    low_confidence_penalty: float = 0.10
    cloud_penalty_weight: float = 0.12

    def to_dict(self) -> dict:
        return asdict(self)


def build_default_policy() -> MissionResponsePolicy:
    return MissionResponsePolicy()
