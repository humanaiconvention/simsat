from __future__ import annotations

from .schemas import EncounterFeatures, EncounterPolicy, EncounterProbeResult, EncounterWindow, TargetSpec


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class FeatureBuilder:
    def pre_rank_score(self, window: EncounterWindow, policy: EncounterPolicy) -> float:
        normalized_peak_elevation = _clamp(
            (window.geometry.elevation_degrees - policy.min_elevation_degrees)
            / max(90.0 - policy.min_elevation_degrees, 1.0)
        )
        return (
            policy.preview_rank_weights["priority"] * window.target_priority
            + policy.preview_rank_weights["elevation"] * normalized_peak_elevation
        )

    def build(
        self,
        window: EncounterWindow,
        target: TargetSpec,
        probe: EncounterProbeResult,
        policy: EncounterPolicy,
    ) -> EncounterFeatures:
        normalized_peak_elevation = _clamp(
            (window.geometry.elevation_degrees - policy.min_elevation_degrees)
            / max(90.0 - policy.min_elevation_degrees, 1.0)
        )
        normalized_duration = _clamp(window.duration_seconds / 120.0)
        sentinel_term = 1.0 if probe.sentinel_available else 0.0
        mapbox_term = 0.0
        cloud_cover_fraction = _clamp((probe.sentinel_cloud_cover or 0.0) / 100.0)
        cloud_penalty = policy.cloud_penalty_weight * cloud_cover_fraction if probe.sentinel_cloud_cover is not None else 0.0

        score_components = {
            "priority": policy.final_score_weights["priority"] * target.priority,
            "elevation": policy.final_score_weights["elevation"] * normalized_peak_elevation,
            "duration": policy.final_score_weights["duration"] * normalized_duration,
            "sentinel": policy.final_score_weights["sentinel"] * sentinel_term,
            "mapbox": policy.final_score_weights["mapbox"] * mapbox_term,
            "cloud_penalty": cloud_penalty,
        }

        return EncounterFeatures(
            window_id=window.window_id,
            target_id=target.target_id,
            duration_seconds=window.duration_seconds,
            peak_elevation_degrees=window.geometry.elevation_degrees,
            off_nadir_degrees=window.geometry.off_nadir_degrees,
            slant_range_km=window.geometry.slant_range_km,
            target_priority=target.priority,
            mapbox_feasible=probe.mapbox_feasible,
            sentinel_available=probe.sentinel_available,
            sentinel_cloud_cover=probe.sentinel_cloud_cover,
            score_components=score_components,
        )
