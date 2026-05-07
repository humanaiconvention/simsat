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


def test_trust_model_requests_refinement_for_compound_medium_risk_case():
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)
    features = EncounterFeatures(
        window_id="window-1",
        target_id="target-1",
        duration_seconds=240.0,
        peak_elevation_degrees=46.0,
        off_nadir_degrees=38.0,
        slant_range_km=700.0,
        target_priority=0.86,
        mapbox_feasible=False,
        sentinel_available=True,
        sentinel_cloud_cover=59.0,
    )

    assessment = model.assess(_window(), features, scaffold_score=0.64)

    assert assessment.trust_band == "medium"
    assert assessment.refinement_reason == "compound_risk_refine"
    assert "edge_geometry" in assessment.reason_codes
    assert "cloud_risk" in assessment.reason_codes


def test_trust_model_online_update_shifts_weights_toward_utility():
    """trust-layer TTT: online_update moves learned_score_weights toward realised utility."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)

    trust_details = {
        "target_priority": 0.9,
        "geometry_margin": 0.85,
        "duration_margin": 1.0,
        "imagery_support": 1.0,
        "clarity_support": 0.9,
        "blend_alpha": 0.65,
    }
    # Operator scores it very useful — learned_score undershot
    weights_before = dict(model._learned_weights)
    updated = model.online_update(trust_details, learned_score=0.65, realized_utility=0.92)

    assert model.update_count == 1
    # Weights must have shifted (not identical to policy defaults)
    assert updated != weights_before
    # Weights must still sum to ~1.0 after renormalisation
    assert abs(sum(updated.values()) - 1.0) < 1e-6
    # All weights remain positive
    assert all(v > 0 for v in updated.values())

    snapshot = model.get_weight_snapshot()
    assert snapshot["update_count"] == 1
    assert len(snapshot["recent_updates"]) == 1
    assert snapshot["recent_updates"][0]["realized_utility"] == 0.92


def test_online_update_l2_regularization_bounds_drift():
    """L2 reg (reg=0.002) pulls weights toward policy defaults, preventing runaway drift."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)
    defaults = dict(policy.learned_score_weights)

    trust_details = {
        "target_priority": 0.95,
        "geometry_margin": 0.95,
        "duration_margin": 0.95,
        "imagery_support": 0.95,
        "clarity_support": 0.95,
    }
    # Run many updates with a consistently positive error (realized > learned)
    for _ in range(200):
        model.online_update(trust_details, learned_score=0.5, realized_utility=0.95)

    weights = model._learned_weights
    # With reg, no weight should diverge beyond 3× its default value
    for k, w in weights.items():
        assert w < defaults[k] * 5.0, f"weight '{k}' drifted to {w:.4f} (default {defaults[k]})"
    # Weights must still sum to 1 after renormalisation
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_online_update_l2_regularization_pulls_back_after_drift():
    """After biased updates, switching to neutral signal lets reg pull weights back toward defaults."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)
    defaults = dict(policy.learned_score_weights)

    trust_details = {
        "target_priority": 0.95,
        "geometry_margin": 0.95,
        "duration_margin": 0.95,
        "imagery_support": 0.95,
        "clarity_support": 0.95,
    }
    # Push weights with 50 biased steps
    for _ in range(50):
        model.online_update(trust_details, learned_score=0.5, realized_utility=0.95)

    drifted = dict(model._learned_weights)

    # Run 500 neutral (error≈0) steps — reg should dominate, pulling weights back
    for _ in range(500):
        model.online_update(trust_details, learned_score=0.70, realized_utility=0.70)

    recovered = model._learned_weights
    for k in defaults:
        assert abs(recovered[k] - defaults[k]) < abs(drifted[k] - defaults[k]) + 0.01, (
            f"weight '{k}' did not recover toward default after neutral signal"
        )


def test_record_skipped_observation_advances_log_without_changing_weights():
    """record_skipped_observation logs the error without altering weights."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)

    trust_details = {
        "target_priority": 0.8,
        "geometry_margin": 0.7,
        "duration_margin": 0.7,
        "imagery_support": 0.8,
        "clarity_support": 0.6,
    }
    weights_before = dict(model._learned_weights)
    count_before = model.update_count

    model.record_skipped_observation(
        trust_details=trust_details,
        learned_score=0.50,
        realized_utility=0.95,
    )

    # Weights must not change
    assert model._learned_weights == weights_before
    # update_count must not increment
    assert model.update_count == count_before
    # But the log must have one new entry tagged blocked=True
    snapshot = model.get_weight_snapshot()
    recent = snapshot["recent_updates"]
    assert len(recent) == 1
    assert recent[0].get("blocked") is True
    assert abs(recent[0]["error"] - (0.95 - 0.50)) < 1e-5


def test_trust_model_online_update_target_priority_in_trust_details():
    """assess() stores target_priority in trust_details so online_update can use it."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)
    features = EncounterFeatures(
        window_id="w1",
        target_id="t1",
        duration_seconds=180.0,
        peak_elevation_degrees=55.0,
        off_nadir_degrees=12.0,
        slant_range_km=560.0,
        target_priority=0.75,
        mapbox_feasible=False,
        sentinel_available=True,
        sentinel_cloud_cover=10.0,
    )
    assessment = model.assess(_window(), features, scaffold_score=0.72)
    assert "target_priority" in assessment.trust_details
    assert assessment.trust_details["target_priority"] == 0.75
