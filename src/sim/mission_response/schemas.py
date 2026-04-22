from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class MissionResponseAction:
    action_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    decision_id: str = ""
    trace_id: str | None = None
    scenario_pack: str = "all"
    target_id: str = ""
    target_label: str = ""
    recommended_action: str = "drop"
    priority: float = 0.0
    utility_score: float = 0.0
    confidence: float = 0.0
    rationale_tags: list[str] = field(default_factory=list)
    status: str = "proposed"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MissionResponseAction":
        return cls(
            action_id=str(raw.get("action_id", f"resp_{uuid.uuid4().hex}")),
            decision_id=str(raw.get("decision_id", "")),
            trace_id=str(raw["trace_id"]) if raw.get("trace_id") is not None else None,
            scenario_pack=str(raw.get("scenario_pack", "all")),
            target_id=str(raw.get("target_id", "")),
            target_label=str(raw.get("target_label", "")),
            recommended_action=str(raw.get("recommended_action", "drop")),
            priority=float(raw.get("priority", 0.0)),
            utility_score=float(raw.get("utility_score", 0.0)),
            confidence=float(raw.get("confidence", 0.0)),
            rationale_tags=[str(v) for v in raw.get("rationale_tags", [])],
            status=str(raw.get("status", "proposed")),
            created_at=str(raw.get("created_at", "")),
        )


@dataclass
class MissionResponseOutcome:
    outcome_id: str = field(default_factory=lambda: f"resp_out_{uuid.uuid4().hex}")
    action_id: str = ""
    executed_at: str = ""
    execution_status: str = "simulated"
    utility_realized: float = 0.0
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MissionResponseOutcome":
        return cls(
            outcome_id=str(raw.get("outcome_id", f"resp_out_{uuid.uuid4().hex}")),
            action_id=str(raw.get("action_id", "")),
            executed_at=str(raw.get("executed_at", "")),
            execution_status=str(raw.get("execution_status", "simulated")),
            utility_realized=float(raw.get("utility_realized", 0.0)),
            notes=str(raw["notes"]) if raw.get("notes") is not None else None,
        )


@dataclass
class MissionUtilitySnapshot:
    mission_id: str = "default"
    scenario_pack: str = "all"
    total_actions: int = 0
    utility_realized: float = 0.0
    utility_missed: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MissionUtilitySnapshot":
        return cls(
            mission_id=str(raw.get("mission_id", "default")),
            scenario_pack=str(raw.get("scenario_pack", "all")),
            total_actions=int(raw.get("total_actions", 0)),
            utility_realized=float(raw.get("utility_realized", 0.0)),
            utility_missed=float(raw.get("utility_missed", 0.0)),
            action_counts={str(k): int(v) for k, v in dict(raw.get("action_counts", {})).items()},
        )
