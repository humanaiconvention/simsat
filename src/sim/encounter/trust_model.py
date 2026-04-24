from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import EncounterFeatures, EncounterPolicy, EncounterWindow

_COMPOUND_REFINE_RISK_CODES = {
    "cloud_risk",
    "edge_geometry",
    "short_margin",
    "no_recent_sentinel_support",
    "scaffold_divergence",
}

_MAX_UPDATE_LOG = 500


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
    """WCLI trust model with online weight adaptation.

    Weights start from policy defaults and drift via `online_update()` as
    operator-reviewed outcomes (realized utility) are fed back at inference time.
    This is the trust-layer TTT loop: no gradient tape, no backward pass — just
    an online least-squares step on the linear learned-score head.
    """

    def __init__(self, policy: EncounterPolicy) -> None:
        self.policy = policy
        # Mutable copies — shift on every online_update call
        self._learned_weights: dict[str, float] = dict(policy.learned_score_weights)
        self._trust_weights: dict[str, float] = dict(policy.trust_score_weights)
        self.update_count: int = 0
        self._update_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

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

        # Use adaptive weights (may have drifted from policy defaults via online_update)
        lw = self._learned_weights
        learned_score = (
            lw["priority"] * features.target_priority
            + lw["geometry"] * geometry_margin
            + lw["duration"] * duration_margin
            + lw["imagery"] * imagery_support
            + lw["clarity"] * clarity_support
        )
        learned_score = _clamp(learned_score)

        tw = self._trust_weights
        agreement = 1.0 - _clamp(abs(learned_score - scaffold_score) / self.policy.agreement_scale)
        trust_score = (
            tw["agreement"] * agreement
            + tw["geometry_margin"] * geometry_margin
            + tw["duration_margin"] * duration_margin
            + tw["imagery_support"] * imagery_support
            + tw["clarity_support"] * clarity_support
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

        refine_risk_count = len([code for code in reason_codes if code in _COMPOUND_REFINE_RISK_CODES])
        if trust_score < self.policy.trust_refine_threshold and scaffold_score >= self.policy.defer_threshold:
            refinement_reason = "trust_below_refine_threshold"
        elif (
            scaffold_score >= self.policy.defer_threshold
            and trust_score < (self.policy.trust_refine_threshold + self.policy.trust_refine_margin)
            and combined_score < self.policy.trust_refine_combined_cap
            and refine_risk_count >= self.policy.trust_compound_risk_min
        ):
            refinement_reason = "compound_risk_refine"

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
                # Included so online_update can reconstruct the full feature vector
                # from a stored trace's trust_details without re-running the probe.
                "target_priority": features.target_priority,
            },
            reason_codes=reason_codes,
            refinement_reason=refinement_reason,
        )

    # ------------------------------------------------------------------
    # Online adaptation (trust-layer TTT)
    # ------------------------------------------------------------------

    def online_update(
        self,
        trust_details: dict[str, float],
        learned_score: float,
        realized_utility: float,
        lr: float = 0.02,
    ) -> dict[str, float]:
        """Online gradient step on learned_score_weights.

        Uses realized utility (operator usefulness_score) as the supervision
        signal.  A single step of online least-squares on the linear learned-
        score head: each weight moves proportionally to its feature activation
        and the prediction error.

        Args:
            trust_details: the trust_details dict stored in a trace's decision_after
                (must include target_priority, geometry_margin, duration_margin,
                imagery_support, clarity_support).
            learned_score: the learned_score value from the same trace.
            realized_utility: operator usefulness_score (0–1).  Pass 0.0 for
                useful=False to push weights away from the current pattern.
            lr: learning rate.  Default 0.02 is intentionally small to prevent
                single-sample instability.

        Returns:
            Updated learned_score_weights dict (also mutates self._learned_weights).
        """
        feature_map = {
            "priority": trust_details.get("target_priority", 0.5),
            "geometry": trust_details.get("geometry_margin", 0.5),
            "duration": trust_details.get("duration_margin", 0.5),
            "imagery": trust_details.get("imagery_support", 0.5),
            "clarity": trust_details.get("clarity_support", 0.5),
        }
        error = realized_utility - learned_score
        for k in list(self._learned_weights):
            self._learned_weights[k] += lr * error * feature_map.get(k, 0.0)
            self._learned_weights[k] = max(0.001, self._learned_weights[k])

        total = sum(self._learned_weights.values())
        self._learned_weights = {k: v / total for k, v in self._learned_weights.items()}

        self.update_count += 1
        entry: dict[str, Any] = {
            "update_idx": self.update_count,
            "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "realized_utility": realized_utility,
            "learned_score": learned_score,
            "error": round(error, 6),
            "weights_after": {k: round(v, 6) for k, v in self._learned_weights.items()},
        }
        self._update_log.append(entry)
        if len(self._update_log) > _MAX_UPDATE_LOG:
            self._update_log = self._update_log[-_MAX_UPDATE_LOG:]

        return dict(self._learned_weights)

    def get_weight_snapshot(self) -> dict[str, Any]:
        """Return current adaptive weights and drift relative to policy defaults."""
        defaults = self.policy.learned_score_weights
        drift = {
            k: round(self._learned_weights.get(k, 0.0) - defaults.get(k, 0.0), 6)
            for k in defaults
        }
        return {
            "learned_score_weights": {k: round(v, 6) for k, v in self._learned_weights.items()},
            "trust_score_weights": {k: round(v, 6) for k, v in self._trust_weights.items()},
            "drift_from_policy_defaults": drift,
            "update_count": self.update_count,
            "policy_id": self.policy.policy_id,
            "recent_updates": self._update_log[-10:],
        }
