from __future__ import annotations


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class MissionUtilityModel:
    def score(
        self,
        target_priority: float,
        effective_combined_score: float,
        confidence: float,
        useful: bool | None,
        cloud_cover: float | None,
    ) -> float:
        cloud_penalty = 0.0 if cloud_cover is None else min(max(cloud_cover / 100.0, 0.0), 1.0) * 0.15
        usefulness_term = 0.0
        if useful is True:
            usefulness_term = 0.08
        elif useful is False:
            usefulness_term = -0.08
        score = (
            0.40 * _clamp(target_priority)
            + 0.40 * _clamp(effective_combined_score)
            + 0.20 * _clamp(confidence)
            + usefulness_term
            - cloud_penalty
        )
        return _clamp(score)
