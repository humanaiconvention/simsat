from __future__ import annotations

from encounter.schemas import EncounterFeatures, EncounterGeometry, EncounterPolicy, EncounterWindow
from encounter.trust_model import WCLITrustModel


def _window() -> EncounterWindow:
    return EncounterWindow(
        window_id="window-1",
        target_id="target-1",
        encounter_type="imaging_window",
        start_time="2026-03-10T12:00:00Z",
        end_time="2026-03-10T12:00:30Z",
        peak_time="2026-03-10T12:00:10Z",
        satellite_position_peak=[0.0, 0.0, 500.0],
        target_position=[0.0, 0.0],
        geometry=EncounterGeometry(
            elevation_degrees=60.0,
            off_nadir_degrees=10.0,
            slant_range_km=550.0,
            target_visible=True,
            bearing=180.0,
            pitch=30.0,
        ),
        target_priority=0.8,
        duration_seconds=30.0,
    )


def test_trust_model_requests_refinement_for_low_trust_case():
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)
    features = EncounterFeatures(
        window_id="window-1",
        target_id="target-1",
        duration_seconds=20.0,
        peak_elevation_degrees=31.0,
        off_nadir_degrees=55.0,
        slant_range_km=900.0,
        target_priority=0.2,
        mapbox_feasible=False,
        sentinel_available=False,
        sentinel_cloud_cover=85.0,
    )

    assessment = model.assess(_window(), features, scaffold_score=0.8)

    assert assessment.trust_band == "low"
    assert assessment.refinement_reason == "trust_below_refine_threshold"
    assert "low_trust" in assessment.reason_codes
    assert "cloud_risk" in assessment.reason_codes
    assert "no_recent_sentinel_support" in assessment.reason_codes


def test_trust_model_keeps_high_confidence_case_out_of_refinement():
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)
    features = EncounterFeatures(
        window_id="window-1",
        target_id="target-1",
        duration_seconds=210.0,
        peak_elevation_degrees=72.0,
        off_nadir_degrees=8.0,
        slant_range_km=520.0,
        target_priority=0.95,
        mapbox_feasible=True,
        sentinel_available=True,
        sentinel_cloud_cover=5.0,
    )

    assessment = model.assess(_window(), features, scaffold_score=0.82)

    assert assessment.trust_band == "high"
    assert assessment.refinement_reason is None
    assert assessment.trust_score >= policy.trust_refine_threshold
    assert "low_trust" not in assessment.reason_codes
