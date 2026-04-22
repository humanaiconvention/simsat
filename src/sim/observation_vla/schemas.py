from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class ObservationImageRef:
    source: str
    image_type: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationImageRef":
        return cls(
            source=str(raw.get("source", "")),
            image_type=str(raw.get("image_type", "")),
            timestamp=str(raw["timestamp"]) if raw.get("timestamp") is not None else None,
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass
class ObservationSample:
    sample_id: str = field(default_factory=lambda: f"obs_{uuid.uuid4().hex}")
    decision_id: str = ""
    window_id: str = ""
    target_id: str = ""
    scenario_pack: str = "all"
    target_label: str = ""
    target_tags: list[str] = field(default_factory=list)
    target_metadata: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)
    images: list[ObservationImageRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["images"] = [image.to_dict() for image in self.images]
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationSample":
        return cls(
            sample_id=str(raw.get("sample_id", f"obs_{uuid.uuid4().hex}")),
            decision_id=str(raw.get("decision_id", "")),
            window_id=str(raw.get("window_id", "")),
            target_id=str(raw.get("target_id", "")),
            scenario_pack=str(raw.get("scenario_pack", "all")),
            target_label=str(raw.get("target_label", "")),
            target_tags=[str(v) for v in raw.get("target_tags", [])],
            target_metadata=dict(raw.get("target_metadata", {})),
            geometry=dict(raw.get("geometry", {})),
            probe=dict(raw.get("probe", {})),
            images=[ObservationImageRef.from_dict(item) for item in raw.get("images", [])],
        )


@dataclass
class ObservationEvidence:
    usable_observation: bool = False
    scene_match_score: float = 0.0
    salience_score: float = 0.0
    change_or_event_score: float = 0.0
    occlusion_or_cloud_risk: float = 0.0
    confidence: float = 0.0
    rationale_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationEvidence":
        return cls(
            usable_observation=bool(raw.get("usable_observation", False)),
            scene_match_score=float(raw.get("scene_match_score", 0.0)),
            salience_score=float(raw.get("salience_score", 0.0)),
            change_or_event_score=float(raw.get("change_or_event_score", 0.0)),
            occlusion_or_cloud_risk=float(raw.get("occlusion_or_cloud_risk", 0.0)),
            confidence=float(raw.get("confidence", 0.0)),
            rationale_tags=[str(v) for v in raw.get("rationale_tags", [])],
        )


@dataclass
class ObservationAssessment:
    assessment_id: str = field(default_factory=lambda: f"asm_{uuid.uuid4().hex}")
    sample_id: str = ""
    model_id: str = ""
    created_at: str = ""
    assessment_mode: str = "metadata_only"
    runtime_mode: str = "stub"
    image_count: int = 0
    confidence_adjustment: float = 0.0
    calibration_tags: list[str] = field(default_factory=list)
    recommended_action: str = "refine"
    evidence: ObservationEvidence = field(default_factory=ObservationEvidence)
    raw_response_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = self.evidence.to_dict()
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationAssessment":
        return cls(
            assessment_id=str(raw.get("assessment_id", f"asm_{uuid.uuid4().hex}")),
            sample_id=str(raw.get("sample_id", "")),
            model_id=str(raw.get("model_id", "")),
            created_at=str(raw.get("created_at", "")),
            assessment_mode=str(raw.get("assessment_mode", "metadata_only")),
            runtime_mode=str(raw.get("runtime_mode", "stub")),
            image_count=int(raw.get("image_count", 0)),
            confidence_adjustment=float(raw.get("confidence_adjustment", 0.0)),
            calibration_tags=[str(v) for v in raw.get("calibration_tags", [])],
            recommended_action=str(raw.get("recommended_action", "refine")),
            evidence=ObservationEvidence.from_dict(raw.get("evidence", {})),
            raw_response_text=str(raw["raw_response_text"]) if raw.get("raw_response_text") is not None else None,
        )


@dataclass
class ObservationResidual:
    assessment_id: str = ""
    trust_delta: float = 0.0
    score_delta: float = 0.0
    recommended_action: str = "refine"
    effective_action: str = "refine"
    rationale_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationResidual":
        return cls(
            assessment_id=str(raw.get("assessment_id", "")),
            trust_delta=float(raw.get("trust_delta", 0.0)),
            score_delta=float(raw.get("score_delta", 0.0)),
            recommended_action=str(raw.get("recommended_action", "refine")),
            effective_action=str(raw.get("effective_action", "refine")),
            rationale_tags=[str(v) for v in raw.get("rationale_tags", [])],
        )


@dataclass
class ObservationOutcome:
    outcome_id: str = field(default_factory=lambda: f"out_{uuid.uuid4().hex}")
    trace_id: str = ""
    assessment_id: str = ""
    decision_id: str = ""
    registered_at: str = ""
    mission_id: str = "default"
    operator_action: str = ""
    useful: bool = False
    usefulness_score: float = 0.0
    label_source: str = "simulated"
    reviewer: str | None = None
    review_status: str = "provisional"
    is_current: bool = True
    supersedes_outcome_id: str | None = None
    outcome_tags: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationOutcome":
        return cls(
            outcome_id=str(raw.get("outcome_id", f"out_{uuid.uuid4().hex}")),
            trace_id=str(raw.get("trace_id", "")),
            assessment_id=str(raw.get("assessment_id", "")),
            decision_id=str(raw.get("decision_id", "")),
            registered_at=str(raw.get("registered_at", "")),
            mission_id=str(raw.get("mission_id", "default")),
            operator_action=str(raw.get("operator_action", "")),
            useful=bool(raw.get("useful", False)),
            usefulness_score=float(raw.get("usefulness_score", 0.0)),
            label_source=str(raw.get("label_source", "simulated")),
            reviewer=str(raw["reviewer"]) if raw.get("reviewer") is not None else None,
            review_status=str(raw.get("review_status", "provisional")),
            is_current=bool(raw.get("is_current", True)),
            supersedes_outcome_id=str(raw["supersedes_outcome_id"]) if raw.get("supersedes_outcome_id") is not None else None,
            outcome_tags=[str(v) for v in raw.get("outcome_tags", [])],
            notes=str(raw["notes"]) if raw.get("notes") is not None else None,
        )


@dataclass
class ObservationMemoryState:
    state_id: str
    scope: str
    scope_id: str
    last_updated: str = ""
    sample_count: int = 0
    useful_count: int = 0
    mean_usefulness: float = 0.0
    confidence_bias: float = 0.0
    refine_bias: float = 0.0
    action_alignment: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationMemoryState":
        return cls(
            state_id=str(raw.get("state_id", "")),
            scope=str(raw.get("scope", "")),
            scope_id=str(raw.get("scope_id", "")),
            last_updated=str(raw.get("last_updated", "")),
            sample_count=int(raw.get("sample_count", 0)),
            useful_count=int(raw.get("useful_count", 0)),
            mean_usefulness=float(raw.get("mean_usefulness", 0.0)),
            confidence_bias=float(raw.get("confidence_bias", 0.0)),
            refine_bias=float(raw.get("refine_bias", 0.0)),
            action_alignment=float(raw.get("action_alignment", 0.0)),
        )


@dataclass
class ObservationAssessmentRecord:
    sample: ObservationSample
    assessment: ObservationAssessment

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "assessment": self.assessment.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationAssessmentRecord":
        return cls(
            sample=ObservationSample.from_dict(raw.get("sample", {})),
            assessment=ObservationAssessment.from_dict(raw.get("assessment", {})),
        )


@dataclass
class ObservationTraceRecord:
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex}")
    created_at: str = ""
    scenario_pack: str = "all"
    decision_id: str = ""
    target_id: str = ""
    window_id: str = ""
    stimulus_id: str | None = None
    outcome_id: str | None = None
    sample: ObservationSample = field(default_factory=ObservationSample)
    assessment: ObservationAssessment = field(default_factory=ObservationAssessment)
    residual: ObservationResidual = field(default_factory=ObservationResidual)
    decision_before: dict[str, Any] = field(default_factory=dict)
    decision_after: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "scenario_pack": self.scenario_pack,
            "decision_id": self.decision_id,
            "target_id": self.target_id,
            "window_id": self.window_id,
            "stimulus_id": self.stimulus_id,
            "outcome_id": self.outcome_id,
            "sample": self.sample.to_dict(),
            "assessment": self.assessment.to_dict(),
            "residual": self.residual.to_dict(),
            "decision_before": self.decision_before,
            "decision_after": self.decision_after,
            "probe": self.probe,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationTraceRecord":
        return cls(
            trace_id=str(raw.get("trace_id", f"trace_{uuid.uuid4().hex}")),
            created_at=str(raw.get("created_at", "")),
            scenario_pack=str(raw.get("scenario_pack", "all")),
            decision_id=str(raw.get("decision_id", "")),
            target_id=str(raw.get("target_id", "")),
            window_id=str(raw.get("window_id", "")),
            stimulus_id=str(raw["stimulus_id"]) if raw.get("stimulus_id") is not None else None,
            outcome_id=str(raw["outcome_id"]) if raw.get("outcome_id") is not None else None,
            sample=ObservationSample.from_dict(raw.get("sample", {})),
            assessment=ObservationAssessment.from_dict(raw.get("assessment", {})),
            residual=ObservationResidual.from_dict(raw.get("residual", {})),
            decision_before=dict(raw.get("decision_before", {})),
            decision_after=dict(raw.get("decision_after", {})),
            probe=dict(raw.get("probe", {})),
            features=dict(raw.get("features", {})),
        )


@dataclass
class SubmissionCasePin:
    scenario_pack: str = "all"
    trace_id: str = ""
    decision_id: str = ""
    target_id: str = ""
    target_label: str = ""
    outcome_id: str = ""
    label_source: str = "operator_review"
    reviewer: str | None = None
    pinned_at: str = ""
    pinned_by: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SubmissionCasePin":
        return cls(
            scenario_pack=str(raw.get("scenario_pack", "all")),
            trace_id=str(raw.get("trace_id", "")),
            decision_id=str(raw.get("decision_id", "")),
            target_id=str(raw.get("target_id", "")),
            target_label=str(raw.get("target_label", "")),
            outcome_id=str(raw.get("outcome_id", "")),
            label_source=str(raw.get("label_source", "operator_review")),
            reviewer=str(raw["reviewer"]) if raw.get("reviewer") is not None else None,
            pinned_at=str(raw.get("pinned_at", "")),
            pinned_by=str(raw["pinned_by"]) if raw.get("pinned_by") is not None else None,
            notes=str(raw["notes"]) if raw.get("notes") is not None else None,
        )
