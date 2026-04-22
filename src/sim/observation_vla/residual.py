from __future__ import annotations

from encounter.schemas import EncounterDecision

from .schemas import ObservationAssessment, ObservationResidual


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ObservationResidualBuilder:
    def build(
        self,
        assessment: ObservationAssessment,
        decision: EncounterDecision,
    ) -> ObservationResidual:
        evidence = assessment.evidence
        scene_support = _clamp(evidence.scene_match_score)
        salience_support = _clamp(evidence.salience_score)
        event_support = _clamp(evidence.change_or_event_score)
        clarity_support = _clamp(1.0 - evidence.occlusion_or_cloud_risk)
        confidence = _clamp(evidence.confidence)

        quality_score = (
            0.35 * scene_support
            + 0.20 * salience_support
            + 0.10 * event_support
            + 0.20 * clarity_support
            + 0.15 * confidence
        )
        mode_scale = 1.0 if assessment.assessment_mode == "image_conditioned" else 0.55

        trust_delta = mode_scale * (
            0.22 * (quality_score - 0.5)
            + 0.18 * (confidence - 0.5)
            + (0.06 if evidence.usable_observation else -0.08)
        )
        score_delta = mode_scale * (
            0.24 * (quality_score - 0.5)
            + (0.05 if assessment.recommended_action == "accept" else 0.0)
            - (0.08 if assessment.recommended_action == "skip" else 0.0)
        )

        trust_delta = _clamp(trust_delta, low=-0.3, high=0.3)
        score_delta = _clamp(score_delta, low=-0.35, high=0.35)

        effective_action = decision.action
        if assessment.recommended_action == "skip" and confidence >= 0.65:
            effective_action = "skip"
        elif (
            assessment.recommended_action == "accept"
            and evidence.usable_observation
            and confidence >= 0.72
            and quality_score >= 0.70
        ):
            effective_action = "accept"
        elif assessment.recommended_action == "defer" and confidence >= 0.55:
            effective_action = "defer"
        elif not evidence.usable_observation and confidence >= 0.6:
            effective_action = "skip"
        elif decision.action in {"accept", "defer"}:
            effective_action = "refine"
        else:
            effective_action = assessment.recommended_action or "refine"

        rationale_tags = list(dict.fromkeys([
            *(evidence.rationale_tags or []),
            f"assessment_mode:{assessment.assessment_mode}",
            f"observation_action:{assessment.recommended_action}",
        ]))

        return ObservationResidual(
            assessment_id=assessment.assessment_id,
            trust_delta=trust_delta,
            score_delta=score_delta,
            recommended_action=assessment.recommended_action,
            effective_action=effective_action,
            rationale_tags=rationale_tags,
        )
