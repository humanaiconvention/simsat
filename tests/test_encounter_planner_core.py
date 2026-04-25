from __future__ import annotations

from encounter.features import FeatureBuilder
from encounter.planner import AnalyticPlanner
from encounter.schemas import EncounterGeometry, EncounterPolicy, EncounterProbeResult, EncounterWindow, TargetSpec
from encounter.trust_model import WCLITrustModel


def _window(*, visible: bool = True, duration_seconds: float = 30.0) -> EncounterWindow:
    return EncounterWindow(
        window_id="window-1",
        target_id="target-1",
        encounter_type="imaging_window",
        start_time="2026-03-10T12:00:00Z",
        end_time="2026-03-10T12:00:20Z",
        peak_time="2026-03-10T12:00:10Z",
        satellite_position_peak=[0.0, 0.0, 500.0],
        target_position=[0.0, 0.0],
        geometry=EncounterGeometry(
            elevation_degrees=68.0 if visible else 10.0,
            off_nadir_degrees=12.0 if visible else 70.0,
            slant_range_km=540.0,
            target_visible=visible,
            bearing=180.0 if visible else None,
            pitch=22.0 if visible else None,
        ),
        target_priority=0.9,
        duration_seconds=duration_seconds,
    )


def _target() -> TargetSpec:
    return TargetSpec(target_id="target-1", label="Target One", lon=0.0, lat=0.0, priority=0.9, size_km=5.0)


def test_planner_core_accepts_high_quality_candidate():
    policy = EncounterPolicy()
    builder = FeatureBuilder()
    planner = AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy))
    probe = EncounterProbeResult(
        window_id="window-1",
        target_id="target-1",
        mapbox_feasible=True,
        sentinel_available=True,
        sentinel_cloud_cover=5.0,
        sentinel_source="sentinel-2a",
        sentinel_datetime="2026-03-10T11:50:00Z",
    )

    features = builder.build(_window(), _target(), probe, policy)
    decision = planner.decide(_window(), features)

    assert decision.action == "accept"
    assert decision.combined_score >= policy.accept_threshold
    assert decision.trust_score >= policy.trust_accept_min
    assert "high_visibility" in decision.reason_codes
    assert "sentinel_available" in decision.reason_codes


def test_planner_core_refines_low_trust_candidate():
    policy = EncounterPolicy()
    builder = FeatureBuilder()
    planner = AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy))
    probe = EncounterProbeResult(
        window_id="window-1",
        target_id="target-1",
        mapbox_feasible=False,
        sentinel_available=False,
        sentinel_cloud_cover=90.0,
        sentinel_source=None,
        sentinel_datetime=None,
    )
    window = EncounterWindow(
        window_id="window-1",
        target_id="target-1",
        encounter_type="imaging_window",
        start_time="2026-03-10T12:00:00Z",
        end_time="2026-03-10T12:01:00Z",
        peak_time="2026-03-10T12:00:30Z",
        satellite_position_peak=[0.0, 0.0, 500.0],
        target_position=[0.0, 0.0],
        geometry=EncounterGeometry(
            elevation_degrees=80.0,
            off_nadir_degrees=55.0,
            slant_range_km=540.0,
            target_visible=True,
            bearing=180.0,
            pitch=22.0,
        ),
        target_priority=1.0,
        duration_seconds=30.0,
    )
    target = TargetSpec(target_id="target-1", label="Target One", lon=0.0, lat=0.0, priority=1.0, size_km=5.0)

    features = builder.build(window, target, probe, policy)
    decision = planner.decide(window, features)

    assert decision.action == "refine"
    assert decision.needs_refinement is True
    assert decision.refinement_reason == "trust_below_refine_threshold"
    assert "low_trust" in decision.reason_codes


def test_planner_core_skips_non_visible_window():
    policy = EncounterPolicy()
    builder = FeatureBuilder()
    planner = AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy))
    window = _window(visible=False, duration_seconds=10.0)
    probe = EncounterProbeResult(
        window_id="window-1",
        target_id="target-1",
        mapbox_feasible=False,
    )

    features = builder.build(window, _target(), probe, policy)
    decision = planner.decide(window, features)

    assert decision.action == "skip"
    assert decision.reason_codes == ["not_visible_or_too_short"]
    assert decision.combined_score == 0.0


def test_planner_core_refines_compound_risk_candidate():
    policy = EncounterPolicy()
    builder = FeatureBuilder()
    planner = AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy))
    probe = EncounterProbeResult(
        window_id="window-1",
        target_id="target-1",
        mapbox_feasible=False,
        sentinel_available=True,
        sentinel_cloud_cover=59.0,
        sentinel_source="sentinel-2a",
        sentinel_datetime="2026-03-10T11:50:00Z",
    )
    window = EncounterWindow(
        window_id="window-1",
        target_id="target-1",
        encounter_type="imaging_window",
        start_time="2026-03-10T12:00:00Z",
        end_time="2026-03-10T12:04:00Z",
        peak_time="2026-03-10T12:02:00Z",
        satellite_position_peak=[0.0, 0.0, 500.0],
        target_position=[0.0, 0.0],
        geometry=EncounterGeometry(
            elevation_degrees=46.0,
            off_nadir_degrees=38.0,
            slant_range_km=700.0,
            target_visible=True,
            bearing=180.0,
            pitch=22.0,
        ),
        target_priority=0.86,
        duration_seconds=240.0,
    )
    target = TargetSpec(target_id="target-1", label="Target One", lon=0.0, lat=0.0, priority=0.86, size_km=5.0)

    features = builder.build(window, target, probe, policy)
    decision = planner.decide(window, features)

    assert decision.action == "refine"
    assert decision.refinement_reason == "compound_risk_refine"
    assert "compound_risk_refine" in decision.reason_codes
