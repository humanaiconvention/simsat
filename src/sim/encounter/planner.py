from __future__ import annotations

from datetime import datetime, timezone

from .schemas import EncounterDecision, EncounterFeatures, EncounterPolicy, EncounterWindow
from .trust_model import WCLITrustModel


class AnalyticPlanner:
    def __init__(self, policy: EncounterPolicy, trust_model: WCLITrustModel | None = None) -> None:
        self.policy = policy
        self.trust_model = trust_model

    def _scaffold_score(self, features: EncounterFeatures) -> float:
        scaffold_score = (
            features.score_components["priority"]
            + features.score_components["elevation"]
            + features.score_components["duration"]
            + features.score_components["sentinel"]
            + features.score_components["mapbox"]
            - features.score_components["cloud_penalty"]
        )
        return max(0.0, min(1.0, scaffold_score))

    def decide(self, window: EncounterWindow, features: EncounterFeatures) -> EncounterDecision:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        if (not window.geometry.target_visible) or window.duration_seconds < self.policy.min_window_seconds:
            return EncounterDecision(
                window_id=window.window_id,
                target_id=window.target_id,
                created_at=created_at,
                policy_version=self.policy.policy_id,
                scaffold_score=0.0,
                learned_score=0.0,
                analytic_score=0.0,
                trust_score=1.0,
                residual_score=0.0,
                combined_score=0.0,
                trust_band="high",
                action="skip",
                reason_codes=["not_visible_or_too_short"],
            )

        scaffold_score = self._scaffold_score(features)

        reason_codes: list[str] = []
        if features.peak_elevation_degrees >= 60.0:
            reason_codes.append("high_visibility")
        if features.target_priority >= 0.8:
            reason_codes.append("high_priority")
        if features.sentinel_available:
            reason_codes.append("sentinel_available")
        if features.sentinel_cloud_cover is not None and features.sentinel_cloud_cover <= 20.0:
            reason_codes.append("low_cloud_risk")
        if window.geometry.target_visible:
            reason_codes.append("line_of_sight")

        learned_score = scaffold_score
        trust_score = 1.0
        residual_score = 0.0
        combined_score = scaffold_score
        trust_band = "high"
        trust_details: dict[str, float] = {}
        refinement_reason: str | None = None

        if self.trust_model is not None:
            assessment = self.trust_model.assess(window, features, scaffold_score)
            learned_score = assessment.learned_score
            trust_score = assessment.trust_score
            residual_score = assessment.residual_score
            combined_score = assessment.combined_score
            trust_band = assessment.trust_band
            trust_details = assessment.trust_details
            refinement_reason = assessment.refinement_reason
            reason_codes.extend(code for code in assessment.reason_codes if code not in reason_codes)

        action = "skip"
        if refinement_reason is not None:
            action = "refine"
            if refinement_reason not in reason_codes:
                reason_codes.append(refinement_reason)
        elif combined_score >= self.policy.accept_threshold and trust_score >= self.policy.trust_accept_min:
            action = "accept"
        elif combined_score >= self.policy.defer_threshold:
            action = "defer"

        return EncounterDecision(
            window_id=window.window_id,
            target_id=window.target_id,
            created_at=created_at,
            policy_version=self.policy.policy_id,
            scaffold_score=scaffold_score,
            learned_score=learned_score,
            analytic_score=scaffold_score,
            trust_score=trust_score,
            residual_score=residual_score,
            combined_score=combined_score,
            trust_band=trust_band,
            trust_details=trust_details,
            action=action,
            reason_codes=reason_codes or [action],
            needs_refinement=action in {"defer", "refine"},
            refinement_reason=refinement_reason,
        )
