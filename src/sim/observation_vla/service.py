from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from encounter.schemas import EncounterRecord, TargetSpec
from haic.schemas import GroundingStimulus
from haic.viability import evaluate_ttt_viability

from .assessor import ObservationAssessor
from .dataset import ObservationDatasetBuilder
from .memory import ObservationTTTMemory
from .residual import ObservationResidualBuilder
from .schemas import (
    ObservationAssessmentRecord,
    ObservationMemoryState,
    ObservationOutcome,
    ObservationResidual,
    ObservationTraceRecord,
    SubmissionCasePin,
)
from .store import ObservationStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from encounter.service import EncounterService
    from mission_response.service import MissionResponseService


class ObservationVLAService:
    def __init__(
        self,
        assessor: ObservationAssessor,
        residual_builder: ObservationResidualBuilder,
        memory: ObservationTTTMemory,
        dataset_builder: ObservationDatasetBuilder,
        store: ObservationStore,
        mission_id: str = "default",
    ) -> None:
        self.assessor = assessor
        self.residual_builder = residual_builder
        self.memory = memory
        self.dataset_builder = dataset_builder
        self.store = store
        self.mission_id = mission_id
        self.encounter_service: "EncounterService | None" = None
        self.mission_response_service: "MissionResponseService | None" = None

    def _sync_mission_response_for_outcome(
        self,
        trace: ObservationTraceRecord,
        outcome: ObservationOutcome,
    ) -> None:
        if self.mission_response_service is None:
            return
        try:
            response = self.mission_response_service.get_trace_response(trace.trace_id)
            action = response.get("action", {})
        except KeyError:
            proposed = self.mission_response_service.propose_from_trace_id(trace.trace_id)
            action = proposed.to_dict() if proposed is not None else {}
        action_id = action.get("action_id")
        if not action_id:
            return
        execution_status = "operator_reviewed" if outcome.label_source == "operator_review" else "simulated"
        notes = (
            f"Synchronized from {outcome.label_source} observation outcome."
            if outcome.label_source
            else "Synchronized from observation outcome."
        )
        self.mission_response_service.register_action_outcome(
            action_id=action_id,
            utility_realized=float(outcome.usefulness_score),
            execution_status=execution_status,
            notes=notes,
        )

    def attach_context(
        self,
        encounter_service: "EncounterService | None" = None,
        mission_response_service: "MissionResponseService | None" = None,
    ) -> None:
        self.encounter_service = encounter_service
        self.mission_response_service = mission_response_service

    def stimulus_has_renderable_image(
        self,
        stimulus: GroundingStimulus | None,
    ) -> bool:
        if stimulus is None:
            return False
        return any(bool(image.image_b64) for image in stimulus.images)

    def is_image_backed_assessment(
        self,
        assessment_record: ObservationAssessmentRecord,
    ) -> bool:
        assessment = assessment_record.assessment
        return (
            assessment.assessment_mode == "image_conditioned"
            and assessment.image_count > 0
            and assessment.runtime_mode not in {"stub", "stub_fallback"}
        )

    def save_assessment(
        self,
        assessment_record: ObservationAssessmentRecord,
    ) -> ObservationAssessmentRecord:
        return self.store.save_record(assessment_record)

    def assess_record(
        self,
        record: EncounterRecord,
        target: TargetSpec,
        stimulus: GroundingStimulus | None = None,
        scenario_pack: str = "all",
        persist: bool = True,
    ) -> ObservationAssessmentRecord:
        sample = self.assessor.build_sample(
            record=record,
            target=target,
            stimulus=stimulus,
            scenario_pack=scenario_pack,
        )
        rendered_images = self.assessor.extract_renderable_images(stimulus)
        assessment = self.assessor.assess(sample, rendered_images=rendered_images)
        assessment = self.memory.apply(sample, assessment, mission_id=self.mission_id)
        stored = ObservationAssessmentRecord(sample=sample, assessment=assessment)
        if persist:
            return self.store.save_record(stored)
        return stored

    def reassess_trace(
        self,
        trace_id: str,
        persist: bool = False,
    ) -> ObservationAssessmentRecord:
        if self.encounter_service is None:
            raise RuntimeError("Encounter service context is not attached")

        trace = self.store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id} not found")

        record = self.encounter_service.store.get_record(trace.decision_id)
        if record is None:
            raise KeyError(f"Encounter record for decision {trace.decision_id} not found")

        target = self.encounter_service.target_repo.get_target(trace.target_id)
        if target is None:
            raise KeyError(f"Target {trace.target_id} not found")

        stimulus = None
        if trace.stimulus_id:
            from haic.stimulus_store import get_stimulus_store

            stimulus = get_stimulus_store().get(trace.stimulus_id)

        sample = self.assessor.build_sample(
            record=record,
            target=target,
            stimulus=stimulus,
            scenario_pack=trace.scenario_pack,
        )
        rendered_images = self.assessor.extract_renderable_images(stimulus)
        assessment = self.assessor.assess(sample, rendered_images=rendered_images)
        assessment = self.memory.apply(sample, assessment, mission_id=self.mission_id)
        assessment_record = ObservationAssessmentRecord(sample=sample, assessment=assessment)
        if persist:
            if not self.is_image_backed_assessment(assessment_record):
                raise RuntimeError(
                    "Observation reassessment did not produce an image-backed model result"
                )
            return self.store.save_record(assessment_record)
        return assessment_record

    def build_residual(
        self,
        assessment_record: ObservationAssessmentRecord,
        decision,
    ) -> ObservationResidual:
        return self.residual_builder.build(assessment_record.assessment, decision)

    def log_trace(
        self,
        scenario_pack: str,
        record: EncounterRecord,
        assessment_record: ObservationAssessmentRecord,
        residual: ObservationResidual,
        decision_before: dict,
        decision_after: dict,
    ) -> ObservationTraceRecord:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        trace = ObservationTraceRecord(
            created_at=created_at,
            scenario_pack=scenario_pack,
            decision_id=record.decision.decision_id,
            target_id=record.decision.target_id,
            window_id=record.window.window_id,
            stimulus_id=record.decision.stimulus_id,
            sample=assessment_record.sample,
            assessment=assessment_record.assessment,
            residual=residual,
            decision_before=decision_before,
            decision_after=decision_after,
            probe=record.probe.to_dict(),
            features=record.features.to_dict(),
        )
        return self.store.save_trace(trace)

    def get_assessment(self, assessment_id: str) -> ObservationAssessmentRecord | None:
        return self.store.get_record(assessment_id)

    def list_assessments(self, limit: int = 50, scenario_pack: str | None = None) -> list[ObservationAssessmentRecord]:
        return self.store.list_records(limit=limit, scenario_pack=scenario_pack)

    def list_traces(self, limit: int = 50, scenario_pack: str | None = None) -> list[ObservationTraceRecord]:
        return self.store.list_traces(limit=limit, scenario_pack=scenario_pack)

    def get_trace(self, trace_id: str) -> ObservationTraceRecord | None:
        return self.store.get_trace(trace_id)

    def get_trace_with_outcome(
        self,
        trace_id: str,
    ) -> tuple[ObservationTraceRecord | None, ObservationOutcome | None]:
        trace = self.store.get_trace(trace_id)
        if trace is None:
            return None, None
        outcome = self.store.get_outcome_for_trace(trace_id)
        return trace, outcome

    def get_current_outcome(self, trace_id: str) -> ObservationOutcome | None:
        return self.store.get_outcome_for_trace(trace_id)

    def get_submission_case(self, scenario_pack: str) -> SubmissionCasePin | None:
        return self.store.get_submission_case(scenario_pack)

    def get_submission_case_by_trace(self, trace_id: str) -> SubmissionCasePin | None:
        return self.store.get_submission_case_by_trace(trace_id)

    def list_submission_cases(self) -> list[SubmissionCasePin]:
        return self.store.list_submission_cases()

    def register_outcome(
        self,
        trace_id: str,
        operator_action: str,
        useful: bool,
        usefulness_score: float,
        outcome_tags: list[str] | None = None,
        notes: str | None = None,
        mission_id: str | None = None,
        label_source: str = "simulated",
        reviewer: str | None = None,
        review_status: str = "provisional",
        replace_existing: bool = True,
    ) -> tuple[ObservationOutcome, list[ObservationMemoryState], ObservationTraceRecord]:
        trace = self.store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id} not found")

        registered_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        existing = self.store.get_outcome_for_trace(trace_id)
        outcome = ObservationOutcome(
            trace_id=trace.trace_id,
            assessment_id=trace.assessment.assessment_id,
            decision_id=trace.decision_id,
            registered_at=registered_at,
            mission_id=mission_id or self.mission_id,
            operator_action=operator_action,
            useful=useful,
            usefulness_score=usefulness_score,
            label_source=label_source,
            reviewer=reviewer,
            review_status=review_status,
            supersedes_outcome_id=existing.outcome_id if existing is not None and replace_existing else None,
            outcome_tags=list(outcome_tags or []),
            notes=notes,
        )
        if existing is not None and replace_existing:
            existing.is_current = False
            existing.review_status = "superseded"
            existing.supersedes_outcome_id = outcome.outcome_id
            self.store.save_outcome(existing)
        saved_outcome = self.store.save_outcome(outcome)
        updated_states = self.memory.rebuild_from_store()
        trace.outcome_id = saved_outcome.outcome_id
        updated_trace = self.store.save_trace(trace)
        if saved_outcome.label_source == "operator_review":
            self._sync_mission_response_for_outcome(updated_trace, saved_outcome)
            self._apply_trust_layer_ttt(updated_trace, saved_outcome)
            self._apply_vla_ttt(updated_trace, saved_outcome)
        return saved_outcome, updated_states, updated_trace

    def _apply_trust_layer_ttt(
        self,
        trace: ObservationTraceRecord,
        outcome: ObservationOutcome,
    ) -> None:
        """Feed realized utility back into the WCLI trust model (trust-layer TTT).

        Pulls trust_details and learned_score from the stored trace's decision_after,
        then calls online_update() on the active trust model.  No-ops silently if the
        encounter service or trust model is not wired.

        Gate behavior:
        - error_bias: BLOCKING — if ≥70% of the last 10 updates share the same error
          sign the adaptation signal is systematically biased; the update is skipped and
          a WARNING is emitted so operators can intervene.
        - weight_drift, update_rate: post-update log-only warnings; they measure
          cumulative drift and cannot block an update already applied.
        """
        if self.encounter_service is None:
            return
        trust_model = getattr(getattr(self.encounter_service, "planner", None), "trust_model", None)
        if trust_model is None:
            return
        decision_after = trace.decision_after or {}
        trust_details = decision_after.get("trust_details")
        learned_score = decision_after.get("learned_score")
        if not trust_details or learned_score is None:
            return
        realized_utility = outcome.usefulness_score if outcome.useful else 0.0
        try:
            # BLOCKING: evaluate error_bias on the pre-update snapshot.
            # Systematic bias in the error window means the adaptation signal is
            # unreliable; skip this update rather than reinforcing the bias.
            pre_snapshot = trust_model.get_weight_snapshot()
            pre_gates = evaluate_ttt_viability(pre_snapshot)
            if not pre_gates.get("error_bias", True):
                logger.warning(
                    "TTT error_bias gate FAILED — update #%d skipped: systematic "
                    "over/under-estimation detected in last 10 updates",
                    pre_snapshot.get("update_count", 0) + 1,
                )
                # Advance the bias window even for blocked updates so the gate
                # can re-evaluate as subsequent operator feedback arrives.
                trust_model.record_skipped_observation(
                    trust_details=trust_details,
                    learned_score=float(learned_score),
                    realized_utility=float(realized_utility),
                )
                return

            trust_model.online_update(
                trust_details=trust_details,
                learned_score=float(learned_score),
                realized_utility=float(realized_utility),
            )
            # Post-update: log-only warnings for weight_drift and update_rate.
            snapshot = trust_model.get_weight_snapshot()
            ttt_gates = evaluate_ttt_viability(snapshot)
            failed_post = [g for g, ok in ttt_gates.items() if not ok and g != "error_bias"]
            if failed_post:
                logger.warning(
                    "TTT viability gate failures after update #%d: %s",
                    snapshot.get("update_count", 0),
                    failed_post,
                )
        except Exception as exc:
            # Never let TTT callback crash the outcome registration; log for triage.
            logger.debug("Trust-layer TTT update raised (suppressed): %s", exc)

    def _apply_vla_ttt(
        self,
        trace: ObservationTraceRecord,
        outcome: ObservationOutcome,
    ) -> None:
        """Feed realized utility back into the VLA adapter's confidence blend weights (VLA-layer TTT).

        Only fires for clip_local runtime traces. Pulls evidence scores from the
        stored assessment and calls vla_online_update() on the active adapter.
        """
        if trace.assessment.runtime_mode != "clip_local":
            return
        adapter = getattr(self, "adapter", None)
        if adapter is None or not hasattr(adapter, "vla_online_update"):
            return
        evidence = trace.assessment.evidence.to_dict()
        confidence = trace.assessment.evidence.confidence
        realized_utility = outcome.usefulness_score if outcome.useful else 0.0
        try:
            adapter.vla_online_update(
                evidence=evidence,
                confidence=confidence,
                realized_utility=float(realized_utility),
            )
        except Exception as exc:
            # Never let VLA online update crash outcome registration; log for triage.
            logger.debug("VLA online update raised (suppressed): %s", exc)

    def register_operator_review(
        self,
        trace_id: str,
        reviewer: str,
        operator_action: str,
        useful: bool,
        usefulness_score: float,
        outcome_tags: list[str] | None = None,
        notes: str | None = None,
        mission_id: str | None = None,
    ) -> tuple[ObservationOutcome, list[ObservationMemoryState], ObservationTraceRecord]:
        tags = list(dict.fromkeys([*(outcome_tags or []), "operator_reviewed"]))
        return self.register_outcome(
            trace_id=trace_id,
            operator_action=operator_action,
            useful=useful,
            usefulness_score=usefulness_score,
            outcome_tags=tags,
            notes=notes,
            mission_id=mission_id,
            label_source="operator_review",
            reviewer=reviewer,
            review_status="reviewed",
            replace_existing=True,
        )

    def list_outcomes(
        self,
        limit: int = 50,
        current_only: bool = True,
        scenario_pack: str | None = None,
    ) -> list[ObservationOutcome]:
        return self.store.list_outcomes(limit=limit, current_only=current_only, scenario_pack=scenario_pack)

    def list_review_candidates(
        self,
        scenario_pack: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        traces = self.store.list_traces(limit=max(limit * 4, limit), scenario_pack=scenario_pack)
        outcomes_by_trace = {
            outcome.trace_id: outcome
            for outcome in self.store.list_outcomes(
                limit=max(limit * 8, limit * 4),
                current_only=True,
                scenario_pack=scenario_pack,
            )
        }
        submission_cases = {
            pin.trace_id: pin
            for pin in self.store.list_submission_cases()
            if scenario_pack in {None, "", "all"} or pin.scenario_pack == scenario_pack
        }
        candidates: list[dict] = []
        for trace in traces:
            outcome = outcomes_by_trace.get(trace.trace_id)
            needs_operator_review = outcome is None or outcome.label_source != "operator_review"
            submission_case = submission_cases.get(trace.trace_id)
            candidates.append(
                {
                    "trace_id": trace.trace_id,
                    "scenario_pack": trace.scenario_pack,
                    "target_id": trace.target_id,
                    "target_label": trace.sample.target_label,
                    "decision_id": trace.decision_id,
                    "assessment_mode": trace.assessment.assessment_mode,
                    "runtime_mode": trace.assessment.runtime_mode,
                    "recommended_action": trace.assessment.recommended_action,
                    "current_outcome": outcome.to_dict() if outcome is not None else None,
                    "needs_operator_review": needs_operator_review,
                    "is_pinned_submission_case": submission_case is not None,
                    "submission_case": submission_case.to_dict() if submission_case is not None else None,
                }
            )
        candidates.sort(
            key=lambda item: (
                1 if item["is_pinned_submission_case"] else 0,
                1 if item["needs_operator_review"] else 0,
                item["current_outcome"]["registered_at"] if item["current_outcome"] is not None else "",
                item["trace_id"],
            ),
            reverse=True,
        )
        return candidates[:limit]

    def get_review_bundle(self, trace_id: str) -> dict[str, Any]:
        trace, outcome = self.get_trace_with_outcome(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id} not found")

        record = None
        target = None
        mission_response = None
        if self.encounter_service is not None:
            record = self.encounter_service.store.get_record(trace.decision_id)
            target = self.encounter_service.target_repo.get_target(trace.target_id)
        if self.mission_response_service is not None:
            try:
                mission_response = self.mission_response_service.get_trace_response(trace_id)
            except KeyError:
                mission_response = None

        submission_case = self.store.get_submission_case_by_trace(trace_id)
        pinned_here = submission_case is not None
        current_label_source = outcome.label_source if outcome is not None else None
        ready_for_submission_case = outcome is not None and outcome.label_source == "operator_review"

        return {
            "trace": trace.to_dict(),
            "current_outcome": outcome.to_dict() if outcome is not None else None,
            "target": target.to_dict() if target is not None else {
                "target_id": trace.target_id,
                "label": trace.sample.target_label,
            },
            "window": record.window.to_dict() if record is not None else {
                "window_id": trace.window_id,
            },
            "planner_decision": record.decision.to_dict() if record is not None else trace.decision_after,
            "probe": record.probe.to_dict() if record is not None else trace.probe,
            "features": record.features.to_dict() if record is not None else trace.features,
            "mission_response": mission_response,
            "submission_case": submission_case.to_dict() if submission_case is not None else None,
            "is_pinned_submission_case": pinned_here,
            "ready_for_operator_review": True,
            "ready_for_submission_case": ready_for_submission_case,
            "label_source": current_label_source,
        }

    def pin_submission_case(
        self,
        scenario_pack: str,
        trace_id: str,
        pinned_by: str | None = None,
        notes: str | None = None,
    ) -> SubmissionCasePin:
        trace, outcome = self.get_trace_with_outcome(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id} not found")
        if trace.scenario_pack != scenario_pack:
            raise ValueError(
                f"Trace {trace_id} belongs to scenario {trace.scenario_pack}, not {scenario_pack}"
            )
        if outcome is None or outcome.label_source != "operator_review":
            raise ValueError("Only traces with a current operator-reviewed outcome can be pinned")

        pinned_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        pin = SubmissionCasePin(
            scenario_pack=scenario_pack,
            trace_id=trace.trace_id,
            decision_id=trace.decision_id,
            target_id=trace.target_id,
            target_label=trace.sample.target_label,
            outcome_id=outcome.outcome_id,
            label_source=outcome.label_source,
            reviewer=outcome.reviewer,
            pinned_at=pinned_at,
            pinned_by=pinned_by,
            notes=notes,
        )
        return self.store.save_submission_case(pin)

    def list_memory_states(self, scope: str | None = None, limit: int = 50) -> list[ObservationMemoryState]:
        return self.store.list_memory_states(scope=scope, limit=limit)

    def export_dataset(self, limit: int = 200, labelled_only: bool = True) -> list[dict]:
        traces = self.store.list_traces(limit=limit)
        outcomes = self.store.list_outcomes(limit=max(limit * 2, limit))
        outcomes_by_trace = {outcome.trace_id: outcome for outcome in outcomes}
        return self.dataset_builder.build_dataset(
            traces,
            outcomes_by_trace_id=outcomes_by_trace,
            labelled_only=labelled_only,
        )
