from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class EphemerisPoint:
    timestamp: str
    satellite_position: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetSpec:
    target_id: str
    label: str
    lon: float
    lat: float
    priority: float = 0.5
    size_km: float = 5.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TargetSpec":
        return cls(
            target_id=str(raw["target_id"]),
            label=str(raw["label"]),
            lon=float(raw["lon"]),
            lat=float(raw["lat"]),
            priority=float(raw.get("priority", 0.5)),
            size_km=float(raw.get("size_km", 5.0)),
            tags=list(raw.get("tags", [])),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass
class EncounterGeometry:
    elevation_degrees: float
    off_nadir_degrees: float
    slant_range_km: float
    target_visible: bool
    bearing: float | None = None
    pitch: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterGeometry":
        return cls(
            elevation_degrees=float(raw.get("elevation_degrees", 0.0)),
            off_nadir_degrees=float(raw.get("off_nadir_degrees", 0.0)),
            slant_range_km=float(raw.get("slant_range_km", 0.0)),
            target_visible=bool(raw.get("target_visible", False)),
            bearing=float(raw["bearing"]) if raw.get("bearing") is not None else None,
            pitch=float(raw["pitch"]) if raw.get("pitch") is not None else None,
        )


@dataclass
class EncounterWindow:
    window_id: str
    target_id: str
    encounter_type: str
    start_time: str
    end_time: str
    peak_time: str
    satellite_position_peak: list[float]
    target_position: list[float]
    geometry: EncounterGeometry
    target_priority: float = 0.5
    duration_seconds: float = 0.0
    pre_rank_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["geometry"] = self.geometry.to_dict()
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterWindow":
        return cls(
            window_id=str(raw["window_id"]),
            target_id=str(raw["target_id"]),
            encounter_type=str(raw.get("encounter_type", "imaging_window")),
            start_time=str(raw["start_time"]),
            end_time=str(raw["end_time"]),
            peak_time=str(raw["peak_time"]),
            satellite_position_peak=[float(v) for v in raw.get("satellite_position_peak", [0.0, 0.0, 0.0])],
            target_position=[float(v) for v in raw.get("target_position", [0.0, 0.0])],
            geometry=EncounterGeometry.from_dict(raw.get("geometry", {})),
            target_priority=float(raw.get("target_priority", 0.5)),
            duration_seconds=float(raw.get("duration_seconds", 0.0)),
            pre_rank_score=float(raw.get("pre_rank_score", 0.0)),
        )


@dataclass
class EncounterProbeResult:
    window_id: str
    target_id: str
    mapbox_feasible: bool
    sentinel_available: bool | None = None
    sentinel_cloud_cover: float | None = None
    sentinel_source: str | None = None
    sentinel_datetime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterProbeResult":
        return cls(
            window_id=str(raw["window_id"]),
            target_id=str(raw["target_id"]),
            mapbox_feasible=bool(raw.get("mapbox_feasible", False)),
            sentinel_available=raw.get("sentinel_available"),
            sentinel_cloud_cover=float(raw["sentinel_cloud_cover"]) if raw.get("sentinel_cloud_cover") is not None else None,
            sentinel_source=str(raw["sentinel_source"]) if raw.get("sentinel_source") is not None else None,
            sentinel_datetime=str(raw["sentinel_datetime"]) if raw.get("sentinel_datetime") is not None else None,
        )


@dataclass
class EncounterFeatures:
    window_id: str
    target_id: str
    duration_seconds: float
    peak_elevation_degrees: float
    off_nadir_degrees: float
    slant_range_km: float
    target_priority: float
    mapbox_feasible: bool
    sentinel_available: bool | None
    sentinel_cloud_cover: float | None
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterFeatures":
        return cls(
            window_id=str(raw["window_id"]),
            target_id=str(raw["target_id"]),
            duration_seconds=float(raw.get("duration_seconds", 0.0)),
            peak_elevation_degrees=float(raw.get("peak_elevation_degrees", 0.0)),
            off_nadir_degrees=float(raw.get("off_nadir_degrees", 0.0)),
            slant_range_km=float(raw.get("slant_range_km", 0.0)),
            target_priority=float(raw.get("target_priority", 0.5)),
            mapbox_feasible=bool(raw.get("mapbox_feasible", False)),
            sentinel_available=raw.get("sentinel_available"),
            sentinel_cloud_cover=float(raw["sentinel_cloud_cover"]) if raw.get("sentinel_cloud_cover") is not None else None,
            score_components={str(k): float(v) for k, v in dict(raw.get("score_components", {})).items()},
        )


@dataclass
class EncounterDecision:
    decision_id: str = field(default_factory=lambda: f"dec_{uuid.uuid4().hex}")
    window_id: str = ""
    target_id: str = ""
    created_at: str = ""
    policy_version: str = ""
    scaffold_score: float = 0.0
    learned_score: float = 0.0
    analytic_score: float = 0.0
    trust_score: float = 1.0
    residual_score: float = 0.0
    combined_score: float = 0.0
    trust_band: str = "high"
    trust_details: dict[str, float] = field(default_factory=dict)
    action: str = "skip"
    reason_codes: list[str] = field(default_factory=list)
    needs_refinement: bool = False
    refinement_reason: str | None = None
    effective_action: str | None = None
    effective_trust_score: float | None = None
    effective_combined_score: float | None = None
    observation_assessment_id: str | None = None
    observation_assessment_mode: str | None = None
    observation_recommended_action: str | None = None
    observation_trust_delta: float = 0.0
    observation_score_delta: float = 0.0
    observation_rationale_tags: list[str] = field(default_factory=list)
    stimulus_id: str | None = None
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterDecision":
        return cls(
            decision_id=str(raw.get("decision_id", f"dec_{uuid.uuid4().hex}")),
            window_id=str(raw.get("window_id", "")),
            target_id=str(raw.get("target_id", "")),
            created_at=str(raw.get("created_at", "")),
            policy_version=str(raw.get("policy_version", "")),
            scaffold_score=float(raw.get("scaffold_score", raw.get("analytic_score", 0.0))),
            learned_score=float(raw.get("learned_score", raw.get("analytic_score", 0.0))),
            analytic_score=float(raw.get("analytic_score", 0.0)),
            trust_score=float(raw.get("trust_score", 1.0)),
            residual_score=float(raw.get("residual_score", 0.0)),
            combined_score=float(raw.get("combined_score", 0.0)),
            trust_band=str(raw.get("trust_band", "high")),
            trust_details={str(k): float(v) for k, v in dict(raw.get("trust_details", {})).items()},
            action=str(raw.get("action", "skip")),
            reason_codes=[str(v) for v in raw.get("reason_codes", [])],
            needs_refinement=bool(raw.get("needs_refinement", False)),
            refinement_reason=str(raw["refinement_reason"]) if raw.get("refinement_reason") is not None else None,
            effective_action=str(raw["effective_action"]) if raw.get("effective_action") is not None else None,
            effective_trust_score=float(raw["effective_trust_score"]) if raw.get("effective_trust_score") is not None else None,
            effective_combined_score=float(raw["effective_combined_score"]) if raw.get("effective_combined_score") is not None else None,
            observation_assessment_id=str(raw["observation_assessment_id"]) if raw.get("observation_assessment_id") is not None else None,
            observation_assessment_mode=str(raw["observation_assessment_mode"]) if raw.get("observation_assessment_mode") is not None else None,
            observation_recommended_action=str(raw["observation_recommended_action"]) if raw.get("observation_recommended_action") is not None else None,
            observation_trust_delta=float(raw.get("observation_trust_delta", 0.0)),
            observation_score_delta=float(raw.get("observation_score_delta", 0.0)),
            observation_rationale_tags=[str(v) for v in raw.get("observation_rationale_tags", [])],
            stimulus_id=str(raw["stimulus_id"]) if raw.get("stimulus_id") is not None else None,
            artifact_id=str(raw["artifact_id"]) if raw.get("artifact_id") is not None else None,
        )


@dataclass
class DecisionArtifact:
    artifact_id: str
    decision_id: str
    created_at: str
    model_id: str
    policy_hash: str
    window_hash: str
    feature_hash: str
    decision_hash: str
    merkle_root: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DecisionArtifact":
        return cls(
            artifact_id=str(raw["artifact_id"]),
            decision_id=str(raw["decision_id"]),
            created_at=str(raw["created_at"]),
            model_id=str(raw["model_id"]),
            policy_hash=str(raw["policy_hash"]),
            window_hash=str(raw["window_hash"]),
            feature_hash=str(raw["feature_hash"]),
            decision_hash=str(raw["decision_hash"]),
            merkle_root=str(raw["merkle_root"]),
            action=str(raw["action"]),
        )


@dataclass
class EncounterRecord:
    decision: EncounterDecision
    window: EncounterWindow
    probe: EncounterProbeResult
    features: EncounterFeatures
    artifact: DecisionArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "decision": self.decision.to_dict(),
            "window": self.window.to_dict(),
            "probe": self.probe.to_dict(),
            "features": self.features.to_dict(),
        }
        if self.artifact is not None:
            data["artifact"] = self.artifact.to_dict()
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterRecord":
        artifact = raw.get("artifact")
        return cls(
            decision=EncounterDecision.from_dict(raw.get("decision", {})),
            window=EncounterWindow.from_dict(raw.get("window", {})),
            probe=EncounterProbeResult.from_dict(raw.get("probe", {})),
            features=EncounterFeatures.from_dict(raw.get("features", {})),
            artifact=DecisionArtifact.from_dict(artifact) if artifact else None,
        )


@dataclass
class EncounterPolicy:
    policy_id: str = "encounter-sentinel-v2"
    min_elevation_degrees: float = 30.0
    min_window_seconds: float = 20.0
    preview_rank_weights: dict[str, float] = field(
        default_factory=lambda: {"priority": 0.6, "elevation": 0.4}
    )
    final_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "priority": 0.35,
            "elevation": 0.30,
            "duration": 0.20,
            "sentinel": 0.15,
            "mapbox": 0.0,
        }
    )
    accept_threshold: float = 0.70
    defer_threshold: float = 0.45
    cloud_penalty_weight: float = 0.15
    scaffold_blend_alpha: float = 0.65
    learned_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "priority": 0.20,
            "geometry": 0.34,
            "duration": 0.18,
            "imagery": 0.16,
            "clarity": 0.12,
        }
    )
    trust_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "agreement": 0.30,
            "geometry_margin": 0.30,
            "duration_margin": 0.15,
            "imagery_support": 0.10,
            "clarity_support": 0.15,
        }
    )
    trust_refine_threshold: float = 0.55
    trust_accept_min: float = 0.45
    agreement_scale: float = 0.35
    residual_scale: float = 0.12
    model_id: str = "wcli-trust-sentinel-v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerEvaluationSummary:
    planner_id: str
    policy_version: str
    model_id: str
    evaluated_windows: int
    action_counts: dict[str, int] = field(default_factory=dict)
    mean_scaffold_score: float = 0.0
    mean_combined_score: float = 0.0
    mean_trust_score: float = 0.0
    materialization_attempts: int = 0
    materialization_successes: int = 0
    materialization_yield: float = 0.0
    top_decision_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlannerEvaluationSummary":
        return cls(
            planner_id=str(raw.get("planner_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            model_id=str(raw.get("model_id", "")),
            evaluated_windows=int(raw.get("evaluated_windows", 0)),
            action_counts={str(k): int(v) for k, v in dict(raw.get("action_counts", {})).items()},
            mean_scaffold_score=float(raw.get("mean_scaffold_score", 0.0)),
            mean_combined_score=float(raw.get("mean_combined_score", 0.0)),
            mean_trust_score=float(raw.get("mean_trust_score", 0.0)),
            materialization_attempts=int(raw.get("materialization_attempts", 0)),
            materialization_successes=int(raw.get("materialization_successes", 0)),
            materialization_yield=float(raw.get("materialization_yield", 0.0)),
            top_decision_ids=[str(v) for v in raw.get("top_decision_ids", [])],
        )


@dataclass
class DecisionDelta:
    window_id: str
    target_id: str
    target_label: str = ""
    scenario_pack: str = "all"
    scaffold_action: str = "skip"
    trust_action: str = "skip"
    scaffold_score: float = 0.0
    trust_score: float = 0.0
    trust_combined_score: float = 0.0
    score_delta: float = 0.0
    action_changed: bool = False
    changed_to_refine: bool = False
    refinement_reason: str | None = None
    scaffold_reason_codes: list[str] = field(default_factory=list)
    trust_reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DecisionDelta":
        return cls(
            window_id=str(raw.get("window_id", "")),
            target_id=str(raw.get("target_id", "")),
            target_label=str(raw.get("target_label", "")),
            scenario_pack=str(raw.get("scenario_pack", "all")),
            scaffold_action=str(raw.get("scaffold_action", "skip")),
            trust_action=str(raw.get("trust_action", "skip")),
            scaffold_score=float(raw.get("scaffold_score", 0.0)),
            trust_score=float(raw.get("trust_score", 0.0)),
            trust_combined_score=float(raw.get("trust_combined_score", 0.0)),
            score_delta=float(raw.get("score_delta", 0.0)),
            action_changed=bool(raw.get("action_changed", False)),
            changed_to_refine=bool(raw.get("changed_to_refine", False)),
            refinement_reason=str(raw["refinement_reason"]) if raw.get("refinement_reason") is not None else None,
            scaffold_reason_codes=[str(v) for v in raw.get("scaffold_reason_codes", [])],
            trust_reason_codes=[str(v) for v in raw.get("trust_reason_codes", [])],
        )


@dataclass
class EncounterEvaluation:
    evaluation_id: str = field(default_factory=lambda: f"eval_{uuid.uuid4().hex}")
    created_at: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    scenario_pack: str = "all"
    window_count: int = 0
    sample_window_ids: list[str] = field(default_factory=list)
    scaffold_summary: PlannerEvaluationSummary | None = None
    trust_summary: PlannerEvaluationSummary | None = None
    action_transition_counts: dict[str, int] = field(default_factory=dict)
    decision_deltas: list[DecisionDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at,
            "parameters": self.parameters,
            "scenario_pack": self.scenario_pack,
            "window_count": self.window_count,
            "sample_window_ids": self.sample_window_ids,
            "action_transition_counts": self.action_transition_counts,
            "decision_deltas": [delta.to_dict() for delta in self.decision_deltas],
        }
        if self.scaffold_summary is not None:
            data["scaffold_summary"] = self.scaffold_summary.to_dict()
        if self.trust_summary is not None:
            data["trust_summary"] = self.trust_summary.to_dict()
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncounterEvaluation":
        scaffold = raw.get("scaffold_summary")
        trust = raw.get("trust_summary")
        return cls(
            evaluation_id=str(raw.get("evaluation_id", f"eval_{uuid.uuid4().hex}")),
            created_at=str(raw.get("created_at", "")),
            parameters=dict(raw.get("parameters", {})),
            scenario_pack=str(raw.get("scenario_pack", "all")),
            window_count=int(raw.get("window_count", 0)),
            sample_window_ids=[str(v) for v in raw.get("sample_window_ids", [])],
            scaffold_summary=PlannerEvaluationSummary.from_dict(scaffold) if scaffold else None,
            trust_summary=PlannerEvaluationSummary.from_dict(trust) if trust else None,
            action_transition_counts={str(k): int(v) for k, v in dict(raw.get("action_transition_counts", {})).items()},
            decision_deltas=[DecisionDelta.from_dict(raw_delta) for raw_delta in raw.get("decision_deltas", [])],
        )


@dataclass
class ScenarioEvidenceCase:
    trace_id: str
    decision_id: str
    target_id: str
    target_label: str = ""
    scenario_pack: str = "all"
    scaffold_action: str = "skip"
    effective_action: str = "skip"
    assessment_mode: str = "metadata_only"
    runtime_mode: str = "stub"
    recommended_action: str = "refine"
    usefulness_score: float | None = None
    useful: bool | None = None
    operator_action: str | None = None
    label_source: str = "simulated"
    reviewer: str | None = None
    review_status: str = "provisional"
    is_pinned_submission_case: bool = False
    cloud_cover: float | None = None
    sentinel_source: str | None = None
    summary_line: str = ""
    evidence_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScenarioEvidenceCase":
        return cls(
            trace_id=str(raw.get("trace_id", "")),
            decision_id=str(raw.get("decision_id", "")),
            target_id=str(raw.get("target_id", "")),
            target_label=str(raw.get("target_label", "")),
            scenario_pack=str(raw.get("scenario_pack", "all")),
            scaffold_action=str(raw.get("scaffold_action", "skip")),
            effective_action=str(raw.get("effective_action", "skip")),
            assessment_mode=str(raw.get("assessment_mode", "metadata_only")),
            runtime_mode=str(raw.get("runtime_mode", "stub")),
            recommended_action=str(raw.get("recommended_action", "refine")),
            usefulness_score=float(raw["usefulness_score"]) if raw.get("usefulness_score") is not None else None,
            useful=bool(raw["useful"]) if raw.get("useful") is not None else None,
            operator_action=str(raw["operator_action"]) if raw.get("operator_action") is not None else None,
            label_source=str(raw.get("label_source", "simulated")),
            reviewer=str(raw["reviewer"]) if raw.get("reviewer") is not None else None,
            review_status=str(raw.get("review_status", "provisional")),
            is_pinned_submission_case=bool(raw.get("is_pinned_submission_case", False)),
            cloud_cover=float(raw["cloud_cover"]) if raw.get("cloud_cover") is not None else None,
            sentinel_source=str(raw["sentinel_source"]) if raw.get("sentinel_source") is not None else None,
            summary_line=str(raw.get("summary_line", "")),
            evidence_tags=[str(v) for v in raw.get("evidence_tags", [])],
        )


@dataclass
class ScenarioEvidenceSummary:
    scenario_pack: str = "all"
    evaluation_id: str | None = None
    created_at: str = ""
    window_count: int = 0
    planner_summary: dict[str, Any] = field(default_factory=dict)
    transition_counts: dict[str, int] = field(default_factory=dict)
    delta_highlights: list[DecisionDelta] = field(default_factory=list)
    labelled_trace_count: int = 0
    unlabelled_trace_count: int = 0
    pinned_submission_trace_id: str | None = None
    reviewed_submission_ready: bool = False
    case_summaries: list[ScenarioEvidenceCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_pack": self.scenario_pack,
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at,
            "window_count": self.window_count,
            "planner_summary": self.planner_summary,
            "transition_counts": self.transition_counts,
            "delta_highlights": [delta.to_dict() for delta in self.delta_highlights],
            "labelled_trace_count": self.labelled_trace_count,
            "unlabelled_trace_count": self.unlabelled_trace_count,
            "pinned_submission_trace_id": self.pinned_submission_trace_id,
            "reviewed_submission_ready": self.reviewed_submission_ready,
            "case_summaries": [case.to_dict() for case in self.case_summaries],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScenarioEvidenceSummary":
        return cls(
            scenario_pack=str(raw.get("scenario_pack", "all")),
            evaluation_id=str(raw["evaluation_id"]) if raw.get("evaluation_id") is not None else None,
            created_at=str(raw.get("created_at", "")),
            window_count=int(raw.get("window_count", 0)),
            planner_summary=dict(raw.get("planner_summary", {})),
            transition_counts={str(k): int(v) for k, v in dict(raw.get("transition_counts", {})).items()},
            delta_highlights=[DecisionDelta.from_dict(item) for item in raw.get("delta_highlights", [])],
            labelled_trace_count=int(raw.get("labelled_trace_count", 0)),
            unlabelled_trace_count=int(raw.get("unlabelled_trace_count", 0)),
            pinned_submission_trace_id=str(raw["pinned_submission_trace_id"]) if raw.get("pinned_submission_trace_id") is not None else None,
            reviewed_submission_ready=bool(raw.get("reviewed_submission_ready", False)),
            case_summaries=[ScenarioEvidenceCase.from_dict(item) for item in raw.get("case_summaries", [])],
        )
