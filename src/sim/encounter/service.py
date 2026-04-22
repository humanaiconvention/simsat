from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from .artifacts import build_artifact
from .ephemeris import EphemerisService
from .features import FeatureBuilder
from .materialize import StimulusMaterializer
from .planner import AnalyticPlanner
from .probes import ImagingProbeService
from .schemas import (
    DecisionDelta,
    EncounterDecision,
    EncounterEvaluation,
    EncounterPolicy,
    EncounterRecord,
    EncounterWindow,
    PlannerEvaluationSummary,
    ScenarioEvidenceCase,
    ScenarioEvidenceSummary,
    TargetSpec,
)
from .store import EncounterStore
from .targets import TargetRepository
from .windows import WindowDetector

if TYPE_CHECKING:
    from observation_vla.service import ObservationVLAService


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class EncounterService:
    def __init__(
        self,
        shared_data: Mapping[str, Any],
        target_repo: TargetRepository,
        ephemeris: EphemerisService,
        window_detector: WindowDetector,
        probe_service: ImagingProbeService,
        feature_builder: FeatureBuilder,
        planner: AnalyticPlanner,
        store: EncounterStore,
        materializer: StimulusMaterializer,
        policy: EncounterPolicy,
        observation_vla: "ObservationVLAService | None" = None,
    ) -> None:
        self.shared_data = shared_data
        self.target_repo = target_repo
        self.ephemeris = ephemeris
        self.window_detector = window_detector
        self.probe_service = probe_service
        self.feature_builder = feature_builder
        self.planner = planner
        self.scaffold_planner = AnalyticPlanner(policy)
        self.store = store
        self.materializer = materializer
        self.policy = policy
        self.observation_vla = observation_vla

    def _resolve_start_time(self, start_time: str | None = None) -> str:
        if start_time:
            return start_time
        shared_start = self.shared_data.get("last_updated") if self.shared_data is not None else None
        if isinstance(shared_start, str) and shared_start:
            return shared_start
        if isinstance(shared_start, (int, float)):
            return datetime.fromtimestamp(float(shared_start), tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _targets_by_id(self) -> dict[str, Any]:
        return {target.target_id: target for target in self.list_targets()}

    def _target_scenario_pack(self, target: TargetSpec) -> str:
        scenario_pack = target.metadata.get("scenario_pack") if isinstance(target.metadata, dict) else None
        if isinstance(scenario_pack, str) and scenario_pack.strip():
            return scenario_pack.strip()
        return "default"

    def list_scenario_packs(self) -> list[str]:
        packs = {self._target_scenario_pack(target) for target in self.list_targets()}
        return sorted(packs)

    def _filter_targets(self, scenario_pack: str | None = None) -> list[TargetSpec]:
        targets = self.list_targets()
        if scenario_pack is None or scenario_pack in {"", "all"}:
            return targets
        return [target for target in targets if self._target_scenario_pack(target) == scenario_pack]

    def _rank_windows(self, windows: list[EncounterWindow]) -> list[EncounterWindow]:
        for window in windows:
            window.pre_rank_score = self.feature_builder.pre_rank_score(window, self.policy)
        return sorted(windows, key=lambda item: item.pre_rank_score, reverse=True)

    def get_policy(self) -> EncounterPolicy:
        return self.policy

    def _preferred_evaluation(
        self,
        evaluations: list[EncounterEvaluation],
    ) -> EncounterEvaluation | None:
        if not evaluations:
            return None
        for evaluation in evaluations:
            if evaluation.window_count > 0:
                return evaluation
        return evaluations[0]

    def list_evaluations(self, limit: int = 20, scenario_pack: str | None = None) -> list[EncounterEvaluation]:
        evaluations = self.store.list_evaluations(limit=None if scenario_pack not in {None, "", "all"} else max(limit * 4, limit))
        if scenario_pack is None or scenario_pack in {"", "all"}:
            return evaluations[:limit]
        filtered = [evaluation for evaluation in evaluations if evaluation.scenario_pack == scenario_pack]
        return filtered[:limit]

    def build_scenario_evidence(
        self,
        scenario_pack: str,
        case_limit: int = 3,
        trace_limit: int = 100,
    ) -> ScenarioEvidenceSummary:
        evaluation = self._preferred_evaluation(self.list_evaluations(limit=20, scenario_pack=scenario_pack))
        if self.observation_vla is None:
            return ScenarioEvidenceSummary(
                scenario_pack=scenario_pack,
                evaluation_id=evaluation.evaluation_id if evaluation else None,
                created_at=evaluation.created_at if evaluation else "",
                window_count=evaluation.window_count if evaluation else 0,
                planner_summary={},
                transition_counts=evaluation.action_transition_counts if evaluation else {},
                delta_highlights=evaluation.decision_deltas[:case_limit] if evaluation else [],
            )

        traces = [
            trace for trace in self.observation_vla.list_traces(limit=trace_limit)
            if trace.scenario_pack == scenario_pack
        ]
        outcomes = self.observation_vla.list_outcomes(limit=max(trace_limit * 2, trace_limit))
        outcomes_by_trace = {outcome.trace_id: outcome for outcome in outcomes if outcome.trace_id}
        labelled_traces = [trace for trace in traces if trace.trace_id in outcomes_by_trace]
        unlabelled_traces = [trace for trace in traces if trace.trace_id not in outcomes_by_trace]
        pinned_submission_case = self.observation_vla.get_submission_case(scenario_pack)

        def _build_case(trace, outcome, *, pinned: bool = False) -> ScenarioEvidenceCase:
            decision_before = trace.decision_before
            decision_after = trace.decision_after
            effective_action = (
                decision_after.get("effective_action")
                or decision_after.get("observation_recommended_action")
                or decision_after.get("action")
                or "skip"
            )
            target = self.target_repo.get_target(trace.target_id)
            target_label = target.label if target is not None else trace.target_id
            cloud_cover = trace.probe.get("sentinel_cloud_cover")
            sentinel_source = trace.probe.get("sentinel_source")
            tags = list(dict.fromkeys([
                *(trace.assessment.evidence.rationale_tags or []),
                *(trace.residual.rationale_tags or []),
                *((outcome.outcome_tags if outcome is not None else []) or []),
                *(["submission_case"] if pinned else []),
            ]))
            summary_line = (
                f"{target_label}: scaffold {decision_before.get('action', 'skip')} -> "
                f"{effective_action}; assessment {trace.assessment.recommended_action}; "
                f"outcome {outcome.operator_action if outcome else 'unlabelled'} "
                f"({'useful' if outcome and outcome.useful else 'not_useful' if outcome else 'pending'})"
            )
            return ScenarioEvidenceCase(
                trace_id=trace.trace_id,
                decision_id=trace.decision_id,
                target_id=trace.target_id,
                target_label=target_label,
                scenario_pack=scenario_pack,
                scaffold_action=str(decision_before.get("action", "skip")),
                effective_action=str(effective_action),
                assessment_mode=trace.assessment.assessment_mode,
                runtime_mode=trace.assessment.runtime_mode,
                recommended_action=trace.assessment.recommended_action,
                usefulness_score=outcome.usefulness_score if outcome is not None else None,
                useful=outcome.useful if outcome is not None else None,
                operator_action=outcome.operator_action if outcome is not None else None,
                label_source=outcome.label_source if outcome is not None else "simulated",
                reviewer=outcome.reviewer if outcome is not None else None,
                review_status=outcome.review_status if outcome is not None else "provisional",
                is_pinned_submission_case=pinned,
                cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
                sentinel_source=str(sentinel_source) if sentinel_source is not None else None,
                summary_line=summary_line,
                evidence_tags=tags,
            )

        preferred_decision_ids = []
        if evaluation is not None:
            preferred_decision_ids.extend(
                delta.window_id for delta in evaluation.decision_deltas if delta.action_changed
            )
        scored_traces = []
        for trace in labelled_traces:
            outcome = outcomes_by_trace.get(trace.trace_id)
            usefulness_score = outcome.usefulness_score if outcome is not None else 0.0
            decision_after = trace.decision_after
            action = (
                decision_after.get("effective_action")
                or decision_after.get("observation_recommended_action")
                or decision_after.get("action")
                or "skip"
            )
            summary_score = (
                1 if outcome is not None and outcome.label_source == "operator_review" else 0,
                usefulness_score,
                1 if action == "refine" else 0,
                1 if trace.window_id in preferred_decision_ids else 0,
                float(decision_after.get("effective_combined_score") or decision_after.get("combined_score") or 0.0),
            )
            scored_traces.append((summary_score, trace, outcome))

        scored_traces.sort(key=lambda item: item[0], reverse=True)
        cases: list[ScenarioEvidenceCase] = []
        pinned_trace_id = None
        reviewed_submission_ready = False
        if pinned_submission_case is not None:
            pinned_trace_id = pinned_submission_case.trace_id
            pinned_trace, pinned_outcome = self.observation_vla.get_trace_with_outcome(pinned_submission_case.trace_id)
            if pinned_trace is not None and pinned_outcome is not None:
                reviewed_submission_ready = pinned_outcome.label_source == "operator_review"
                cases.append(_build_case(pinned_trace, pinned_outcome, pinned=True))

        for _, trace, outcome in scored_traces:
            if len(cases) >= case_limit:
                break
            if trace.trace_id == pinned_trace_id:
                continue
            cases.append(_build_case(trace, outcome, pinned=False))

        planner_summary = {}
        if evaluation is not None and evaluation.scaffold_summary is not None and evaluation.trust_summary is not None:
            planner_summary = {
                "scaffold_actions": evaluation.scaffold_summary.action_counts,
                "trust_actions": evaluation.trust_summary.action_counts,
                "scaffold_yield": evaluation.scaffold_summary.materialization_yield,
                "trust_yield": evaluation.trust_summary.materialization_yield,
            }

        return ScenarioEvidenceSummary(
            scenario_pack=scenario_pack,
            evaluation_id=evaluation.evaluation_id if evaluation else None,
            created_at=evaluation.created_at if evaluation else "",
            window_count=evaluation.window_count if evaluation else 0,
            planner_summary=planner_summary,
            transition_counts=evaluation.action_transition_counts if evaluation else {},
            delta_highlights=evaluation.decision_deltas[:case_limit] if evaluation else [],
            labelled_trace_count=len(labelled_traces),
            unlabelled_trace_count=len(unlabelled_traces),
            pinned_submission_trace_id=pinned_trace_id,
            reviewed_submission_ready=reviewed_submission_ready,
            case_summaries=cases,
        )

    def list_targets(self) -> list[TargetSpec]:
        return self.target_repo.list_targets()

    def list_targets_for_scenario(self, scenario_pack: str | None = None) -> list[TargetSpec]:
        return self._filter_targets(scenario_pack=scenario_pack)

    def create_target(self, target: TargetSpec) -> TargetSpec:
        if self.target_repo.get_target(target.target_id) is not None:
            raise ValueError(f"Target {target.target_id} already exists")
        return self.target_repo.upsert_target(target)

    def update_target(self, target_id: str, target: TargetSpec) -> TargetSpec:
        if target_id != target.target_id:
            raise ValueError("Target ID in path and body must match")
        if self.target_repo.get_target(target_id) is None:
            raise KeyError(f"Target {target_id} not found")
        return self.target_repo.upsert_target(target)

    def delete_target(self, target_id: str) -> bool:
        return self.target_repo.delete_target(target_id)

    def preview_windows(
        self,
        start_time: str | None = None,
        hours: float = 6.0,
        step_seconds: int = 30,
        top_k: int = 20,
        scenario_pack: str | None = None,
    ) -> list[EncounterWindow]:
        targets = self._filter_targets(scenario_pack=scenario_pack)
        ephemeris_points = self.ephemeris.propagate_series(
            self._resolve_start_time(start_time),
            hours=hours,
            step_seconds=step_seconds,
        )
        windows = self.window_detector.enumerate_windows(ephemeris_points, targets)
        return self._rank_windows(windows)[:top_k]

    def _build_record(
        self,
        decision: EncounterDecision,
        window: EncounterWindow,
        probe,
        features,
    ) -> EncounterRecord:
        return EncounterRecord(
            decision=decision,
            window=window,
            probe=probe,
            features=features,
        )

    def _build_decision_delta(
        self,
        target: TargetSpec,
        window: EncounterWindow,
        scaffold_decision: EncounterDecision,
        trust_decision: EncounterDecision,
    ) -> DecisionDelta:
        return DecisionDelta(
            window_id=window.window_id,
            target_id=window.target_id,
            target_label=target.label,
            scenario_pack=self._target_scenario_pack(target),
            scaffold_action=scaffold_decision.action,
            trust_action=trust_decision.action,
            scaffold_score=scaffold_decision.combined_score,
            trust_score=trust_decision.trust_score,
            trust_combined_score=trust_decision.combined_score,
            score_delta=trust_decision.combined_score - scaffold_decision.combined_score,
            action_changed=scaffold_decision.action != trust_decision.action,
            changed_to_refine=trust_decision.action == "refine" and scaffold_decision.action != "refine",
            refinement_reason=trust_decision.refinement_reason,
            scaffold_reason_codes=list(scaffold_decision.reason_codes),
            trust_reason_codes=list(trust_decision.reason_codes),
        )

    def _rank_decision_deltas(self, deltas: list[DecisionDelta], limit: int = 8) -> list[DecisionDelta]:
        ranked = sorted(
            deltas,
            key=lambda delta: (
                1 if delta.changed_to_refine else 0,
                1 if delta.action_changed else 0,
                abs(delta.score_delta),
                delta.trust_combined_score,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _planner_materialization_yield(
        self,
        planner_records: list[EncounterRecord],
        targets_by_id: dict[str, TargetSpec],
        materialize_top_k: int,
    ) -> tuple[int, int, list[str]]:
        if materialize_top_k <= 0:
            return 0, 0, []

        candidates = [
            record for record in planner_records
            if record.decision.action in {"accept", "defer", "refine"}
        ]
        candidates.sort(key=lambda record: record.decision.combined_score, reverse=True)
        selected = candidates[:materialize_top_k]

        attempts = 0
        successes = 0
        attempted_ids: list[str] = []
        for record in selected:
            target = targets_by_id.get(record.decision.target_id)
            if target is None:
                continue
            attempts += 1
            attempted_ids.append(record.decision.decision_id)
            try:
                stimulus = self.materializer.materialize(record, target, persist=False)
                has_imagery = any(
                    (image.image_b64 is not None)
                    or (image.metadata is not None and image.metadata.image_available)
                    for image in stimulus.images
                )
                if has_imagery:
                    successes += 1
            except Exception:
                continue
        return attempts, successes, attempted_ids

    def _summarize_planner(
        self,
        planner_id: str,
        model_id: str,
        records: list[EncounterRecord],
        targets_by_id: dict[str, TargetSpec],
        materialize_top_k: int,
    ) -> PlannerEvaluationSummary:
        action_counts = {"accept": 0, "defer": 0, "refine": 0, "skip": 0}
        evaluated_windows = len(records)
        scaffold_sum = 0.0
        combined_sum = 0.0
        trust_sum = 0.0

        for record in records:
            action_counts.setdefault(record.decision.action, 0)
            action_counts[record.decision.action] += 1
            scaffold_sum += record.decision.scaffold_score
            combined_sum += record.decision.combined_score
            trust_sum += record.decision.trust_score

        attempts, successes, top_decision_ids = self._planner_materialization_yield(
            records,
            targets_by_id,
            materialize_top_k,
        )
        denom = evaluated_windows or 1
        return PlannerEvaluationSummary(
            planner_id=planner_id,
            policy_version=self.policy.policy_id,
            model_id=model_id,
            evaluated_windows=evaluated_windows,
            action_counts=action_counts,
            mean_scaffold_score=scaffold_sum / denom,
            mean_combined_score=combined_sum / denom,
            mean_trust_score=trust_sum / denom,
            materialization_attempts=attempts,
            materialization_successes=successes,
            materialization_yield=(successes / attempts) if attempts else 0.0,
            top_decision_ids=top_decision_ids,
        )

    def evaluate(
        self,
        start_time: str | None = None,
        hours: float = 6.0,
        step_seconds: int = 30,
        top_k: int = 20,
        materialize_top_k: int = 3,
        scenario_pack: str | None = None,
    ) -> EncounterEvaluation:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        windows = self.preview_windows(
            start_time=start_time,
            hours=hours,
            step_seconds=step_seconds,
            top_k=top_k,
            scenario_pack=scenario_pack,
        )
        targets_by_id = self._targets_by_id()

        scaffold_records: list[EncounterRecord] = []
        trust_records: list[EncounterRecord] = []
        transition_counts: dict[str, int] = {}
        decision_deltas: list[DecisionDelta] = []

        for window in windows:
            target = targets_by_id.get(window.target_id)
            if target is None:
                continue
            probe = self.probe_service.probe(window, target)
            features = self.feature_builder.build(window, target, probe, self.policy)

            scaffold_decision = self.scaffold_planner.decide(window, features)
            trust_decision = self.planner.decide(window, features)

            scaffold_records.append(self._build_record(scaffold_decision, window, probe, features))
            trust_records.append(self._build_record(trust_decision, window, probe, features))
            decision_deltas.append(
                self._build_decision_delta(
                    target=target,
                    window=window,
                    scaffold_decision=scaffold_decision,
                    trust_decision=trust_decision,
                )
            )

            transition_key = f"{scaffold_decision.action}->{trust_decision.action}"
            transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1

        evaluation = EncounterEvaluation(
            created_at=created_at,
            parameters={
                "start_time": self._resolve_start_time(start_time),
                "hours": hours,
                "step_seconds": step_seconds,
                "top_k": top_k,
                "materialize_top_k": materialize_top_k,
                "scenario_pack": scenario_pack or "all",
            },
            scenario_pack=scenario_pack or "all",
            window_count=len(windows),
            sample_window_ids=[window.window_id for window in windows[: min(10, len(windows))]],
            scaffold_summary=self._summarize_planner(
                planner_id="scaffold",
                model_id="analytic-scaffold-v1",
                records=scaffold_records,
                targets_by_id=targets_by_id,
                materialize_top_k=materialize_top_k,
            ),
            trust_summary=self._summarize_planner(
                planner_id="wcli_trust",
                model_id=self.policy.model_id,
                records=trust_records,
                targets_by_id=targets_by_id,
                materialize_top_k=materialize_top_k,
            ),
            action_transition_counts=transition_counts,
            decision_deltas=self._rank_decision_deltas(decision_deltas),
        )
        self.store.save_evaluation(evaluation)
        return evaluation

    def plan(
        self,
        start_time: str | None = None,
        hours: float = 6.0,
        step_seconds: int = 30,
        top_k: int = 20,
        scenario_pack: str | None = None,
    ) -> list[EncounterRecord]:
        records: list[EncounterRecord] = []
        targets_by_id = self._targets_by_id()
        for window in self.preview_windows(
            start_time=start_time,
            hours=hours,
            step_seconds=step_seconds,
            top_k=top_k,
            scenario_pack=scenario_pack,
        ):
            target = targets_by_id.get(window.target_id)
            if target is None:
                continue
            probe = self.probe_service.probe(window, target)
            features = self.feature_builder.build(window, target, probe, self.policy)
            decision = self.planner.decide(window, features)
            record = self._build_record(decision, window, probe, features)
            artifact = build_artifact(record, self.policy)
            record.artifact = artifact
            record.decision.artifact_id = artifact.artifact_id
            self.store.save_record(record)
            records.append(record)
        return records

    def list_decisions(self, limit: int = 50) -> list[EncounterDecision]:
        return [record.decision for record in self.store.list_records(limit=limit)]

    def materialize_decision(self, decision_id: str) -> EncounterRecord:
        record = self.store.get_record(decision_id)
        if record is None:
            raise KeyError(f"Decision {decision_id} not found")
        target = self.target_repo.get_target(record.decision.target_id)
        if target is None:
            raise KeyError(f"Target {record.decision.target_id} not found")
        stimulus = self.materializer.materialize(record, target)
        record.decision.stimulus_id = stimulus.stimulus_id
        self.store.save_record(record)
        return record

    def _apply_observation_residual(
        self,
        record: EncounterRecord,
        assessment_record,
        residual,
    ) -> EncounterRecord:
        decision = record.decision
        decision.observation_assessment_id = assessment_record.assessment.assessment_id
        decision.observation_assessment_mode = assessment_record.assessment.assessment_mode
        decision.observation_recommended_action = residual.recommended_action
        decision.observation_trust_delta = residual.trust_delta
        decision.observation_score_delta = residual.score_delta
        decision.observation_rationale_tags = list(residual.rationale_tags)
        decision.effective_action = residual.effective_action
        decision.effective_trust_score = _clamp(decision.trust_score + residual.trust_delta)
        decision.effective_combined_score = _clamp(decision.combined_score + residual.score_delta)
        decision.needs_refinement = decision.effective_action in {"defer", "refine"}

        updated_reason_codes = list(dict.fromkeys([
            *decision.reason_codes,
            *(f"observation:{tag}" for tag in residual.rationale_tags),
        ]))
        decision.reason_codes = updated_reason_codes
        self.store.save_record(record)
        return record

    def assess_materialized_decision(self, decision_id: str) -> dict[str, Any]:
        if self.observation_vla is None:
            raise RuntimeError("Observation VLA service is not configured")

        record = self.store.get_record(decision_id)
        if record is None:
            raise KeyError(f"Decision {decision_id} not found")

        target = self.target_repo.get_target(record.decision.target_id)
        if target is None:
            raise KeyError(f"Target {record.decision.target_id} not found")

        stimulus = None
        if record.decision.stimulus_id:
            from haic.stimulus_store import get_stimulus_store

            stimulus = get_stimulus_store().get(record.decision.stimulus_id)

        if stimulus is None:
            record = self.materialize_decision(decision_id)
            from haic.stimulus_store import get_stimulus_store

            stimulus = get_stimulus_store().get(record.decision.stimulus_id or "")

        if not self.observation_vla.stimulus_has_renderable_image(stimulus):
            raise RuntimeError(
                "Materialized stimulus has no renderable image payload; refusing to persist a metadata-only observation trace"
            )

        scenario_pack = self._target_scenario_pack(target)
        decision_before = deepcopy(record.decision.to_dict())
        assessment_record = self.observation_vla.assess_record(
            record=record,
            target=target,
            stimulus=stimulus,
            scenario_pack=scenario_pack,
            persist=False,
        )
        if not self.observation_vla.is_image_backed_assessment(assessment_record):
            raise RuntimeError(
                f"Observation VLA fell back to {assessment_record.assessment.runtime_mode}; refusing to persist a low-value fallback trace"
            )
        self.observation_vla.save_assessment(assessment_record)
        residual = self.observation_vla.build_residual(assessment_record, record.decision)
        record = self._apply_observation_residual(record, assessment_record, residual)
        trace = self.observation_vla.log_trace(
            scenario_pack=scenario_pack,
            record=record,
            assessment_record=assessment_record,
            residual=residual,
            decision_before=decision_before,
            decision_after=record.decision.to_dict(),
        )
        return {
            "decision_id": record.decision.decision_id,
            "effective_action": record.decision.effective_action or record.decision.action,
            "effective_trust_score": record.decision.effective_trust_score,
            "effective_combined_score": record.decision.effective_combined_score,
            "residual": residual.to_dict(),
            "trace_id": trace.trace_id,
            **assessment_record.to_dict(),
            "materialized_stimulus_found": stimulus is not None,
            "assessment_mode": assessment_record.assessment.assessment_mode,
        }
