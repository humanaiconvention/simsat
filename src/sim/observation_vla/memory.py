from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from .schemas import (
    ObservationAssessment,
    ObservationMemoryState,
    ObservationOutcome,
    ObservationSample,
    ObservationTraceRecord,
)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ObservationTTTMemory:
    def __init__(self, store) -> None:
        self.store = store

    def _now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _get_state(self, scope: str, scope_id: str) -> ObservationMemoryState:
        existing = self.store.get_memory_state(scope, scope_id)
        if existing is not None:
            return existing
        return ObservationMemoryState(
            state_id=f"{scope}:{scope_id}",
            scope=scope,
            scope_id=scope_id,
            last_updated=self._now(),
        )

    def _combined_biases(self, sample: ObservationSample, mission_id: str) -> tuple[float, float, list[str]]:
        states = [
            ("pass", sample.window_id, 0.15),
            ("scenario", sample.scenario_pack or "all", 0.5),
            ("mission", mission_id or "default", 0.35),
        ]
        confidence_adjustment = 0.0
        refine_bias = 0.0
        tags: list[str] = []
        for scope, scope_id, weight in states:
            state = self.store.get_memory_state(scope, scope_id)
            if state is None or state.sample_count == 0:
                continue
            confidence_adjustment += weight * state.confidence_bias
            refine_bias += weight * state.refine_bias
            tags.append(f"{scope}_ttt:{scope_id}")
        return _clamp(confidence_adjustment, low=-0.25, high=0.25), _clamp(refine_bias, low=-0.3, high=0.3), tags

    def apply(
        self,
        sample: ObservationSample,
        assessment: ObservationAssessment,
        mission_id: str = "default",
    ) -> ObservationAssessment:
        adjusted = deepcopy(assessment)
        confidence_adjustment, refine_bias, tags = self._combined_biases(sample, mission_id=mission_id)
        adjusted.confidence_adjustment = confidence_adjustment
        adjusted.calibration_tags = list(dict.fromkeys([*adjusted.calibration_tags, *tags]))
        adjusted.evidence.confidence = _clamp(
            adjusted.evidence.confidence + confidence_adjustment,
            low=0.0,
            high=1.0,
        )

        if adjusted.recommended_action == "accept" and (
            (
                adjusted.evidence.confidence < 0.60
                and adjusted.evidence.scene_match_score < 0.62
            )
            or refine_bias > 0.18
        ):
            adjusted.recommended_action = "refine"
            adjusted.calibration_tags.append("ttt_downgraded_accept")
        elif adjusted.recommended_action == "refine" and (
            adjusted.evidence.usable_observation
            and adjusted.evidence.confidence >= 0.67
            and adjusted.evidence.scene_match_score >= 0.58
            and adjusted.evidence.occlusion_or_cloud_risk <= 0.38
            and refine_bias < -0.08
        ):
            adjusted.recommended_action = "accept"
            adjusted.calibration_tags.append("ttt_upgraded_refine")
        elif adjusted.recommended_action == "skip" and (
            adjusted.evidence.usable_observation and adjusted.evidence.confidence >= 0.62 and confidence_adjustment > 0.08
        ):
            adjusted.recommended_action = "defer"
            adjusted.calibration_tags.append("ttt_recovered_skip")

        adjusted.calibration_tags = list(dict.fromkeys(adjusted.calibration_tags))
        return adjusted

    def register_outcome(
        self,
        trace: ObservationTraceRecord,
        outcome: ObservationOutcome,
    ) -> list[ObservationMemoryState]:
        states_by_key: dict[tuple[str, str], ObservationMemoryState] = {}
        return self._register_outcome(trace, outcome, states_by_key)

    def _register_outcome(
        self,
        trace: ObservationTraceRecord,
        outcome: ObservationOutcome,
        states_by_key: dict[tuple[str, str], ObservationMemoryState],
    ) -> list[ObservationMemoryState]:
        predicted_action = (
            trace.decision_after.get("effective_action")
            or trace.assessment.recommended_action
            or trace.decision_after.get("action")
            or "refine"
        )
        predicted_confidence = float(trace.assessment.evidence.confidence)
        target_confidence = 0.8 if outcome.useful else 0.35
        confidence_error = _clamp(target_confidence - predicted_confidence, low=-0.4, high=0.4)

        optimistic_error = predicted_action in {"accept", "defer"} and not outcome.useful
        conservative_error = predicted_action in {"refine", "skip"} and outcome.useful and outcome.operator_action == "accept"
        refine_shift = 0.0
        if optimistic_error:
            refine_shift = 0.12
        elif conservative_error:
            refine_shift = -0.1

        alignment = 1.0 if outcome.operator_action == predicted_action else 0.0
        updated_states: list[ObservationMemoryState] = []
        scope_items = [
            ("pass", trace.window_id),
            ("scenario", trace.scenario_pack or "all"),
            ("mission", outcome.mission_id or "default"),
        ]
        for scope, scope_id in scope_items:
            state = states_by_key.get((scope, scope_id))
            if state is None:
                state = self._get_state(scope, scope_id)
            prev_count = state.sample_count
            state.sample_count += 1
            state.useful_count += int(outcome.useful)
            state.mean_usefulness = (
                ((state.mean_usefulness * prev_count) + outcome.usefulness_score) / state.sample_count
            )
            state.confidence_bias = _clamp(
                (state.confidence_bias * 0.8) + (confidence_error * 0.2),
                low=-0.25,
                high=0.25,
            )
            state.refine_bias = _clamp(
                (state.refine_bias * 0.8) + (refine_shift * 0.2),
                low=-0.3,
                high=0.3,
            )
            state.action_alignment = (
                ((state.action_alignment * prev_count) + alignment) / state.sample_count
            )
            state.last_updated = self._now()
            states_by_key[(scope, scope_id)] = state
            updated_states.append(state)
        return updated_states

    def rebuild_from_store(self) -> list[ObservationMemoryState]:
        traces = self.store.list_traces(limit=10000)
        traces_by_id = {trace.trace_id: trace for trace in traces}
        outcomes = self.store.list_outcomes(limit=10000, current_only=True)
        outcomes.sort(key=lambda outcome: outcome.registered_at)

        states_by_key: dict[tuple[str, str], ObservationMemoryState] = {}
        for outcome in outcomes:
            trace = traces_by_id.get(outcome.trace_id)
            if trace is None:
                continue
            self._register_outcome(trace, outcome, states_by_key)

        states = sorted(
            states_by_key.values(),
            key=lambda state: (state.scope, state.scope_id),
        )
        self.store.replace_memory_states(states)
        return states
