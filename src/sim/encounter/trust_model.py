from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import EncounterFeatures, EncounterPolicy, EncounterWindow


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class TrustAssessment:
    scaffold_score: float
    learned_score: float
    trust_score: float
    residual_score: float
    combined_score: float
    trust_band: str
    trust_details: dict[str, float] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    refinement_reason: str | None = None


class WCLITrustModel:
    def __init__(self, policy: EncounterPolicy) -> None:
        self.policy = policy

    def assess(self, window: EncounterWindow, features: EncounterFeatures, scaffold_score: float) -> TrustAssessment:
        elevation_margin = _clamp(
            (features.peak_elevation_degrees - self.policy.min_elevation_degrees) / 30.0
        )
        duration_margin = _clamp(
            (features.duration_seconds - self.policy.min_window_seconds) / 180.0
        )
        geometry_margin = _clamp(1.0 - (features.off_nadir_degrees / 60.0))
        imagery_support = 1.0 if features.sentinel_available else (0.35 if window.geometry.target_visible else 0.15)
        clarity_support = (
            1.0 - _clamp((features.sentinel_cloud_cover or 0.0) / 100.0)
            if features.sentinel_cloud_cover is not None
            else 0.35
        )

        learned_score = (
            self.policy.learned_score_weights["priority"] * features.target_priority
            + self.policy.learned_score_weights["geometry"] * geometry_margin
            + self.policy.learned_score_weights["duration"] * duration_margin
            + self.policy.learned_score_weights["imagery"] * imagery_support
            + self.policy.learned_score_weights["clarity"] * clarity_support
        )
        learned_score = _clamp(learned_score)

        agreement = 1.0 - _clamp(abs(learned_score - scaffold_score) / self.policy.agreement_scale)
        trust_score = (
            self.policy.trust_score_weights["agreement"] * agreement
            + self.policy.trust_score_weights["geometry_margin"] * geometry_margin
            + self.policy.trust_score_weights["duration_margin"] * duration_margin
            + self.policy.trust_score_weights["imagery_support"] * imagery_support
            + self.policy.trust_score_weights["clarity_support"] * clarity_support
        )
        trust_score = _clamp(trust_score)

        residual_signal = _clamp(learned_score - scaffold_score, low=-1.0, high=1.0)
        blended_score = (
            self.policy.scaffold_blend_alpha * scaffold_score
            + (1.0 - self.policy.scaffold_blend_alpha) * learned_score
        )
        residual_score = self.policy.residual_scale * trust_score * residual_signal
        combined_score = _clamp(blended_score + residual_score)

        if trust_score >= 0.75:
            trust_band = "high"
        elif trust_score >= 0.5:
            trust_band = "medium"
        else:
            trust_band = "low"

        reason_codes: list[str] = []
        refinement_reason: str | None = None
        if trust_band == "low":
            reason_codes.append("low_trust")
        if agreement < 0.5:
            reason_codes.append("scaffold_divergence")
        if geometry_margin < 0.45:
            reason_codes.append("edge_geometry")
        if duration_margin < 0.35:
            reason_codes.append("short_margin")
        if not features.sentinel_available:
            reason_codes.append("no_recent_sentinel_support")
        if features.sentinel_cloud_cover is not None and features.sentinel_cloud_cover > 40.0:
            reason_codes.append("cloud_risk")
        if trust_score < self.policy.trust_refine_threshold and scaffold_score >= self.policy.defer_threshold:
            refinement_reason = "trust_below_refine_threshold"

        return TrustAssessment(
            scaffold_score=scaffold_score,
            learned_score=learned_score,
            trust_score=trust_score,
            residual_score=residual_score,
            combined_score=combined_score,
            trust_band=trust_band,
            trust_details={
                "agreement": agreement,
                "geometry_margin": geometry_margin,
                "duration_margin": duration_margin,
                "imagery_support": imagery_support,
                "clarity_support": clarity_support,
                "blend_alpha": self.policy.scaffold_blend_alpha,
            },
            reason_codes=reason_codes,
            refinement_reason=refinement_reason,
        )
