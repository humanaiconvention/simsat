from __future__ import annotations

import pytest

from encounter.schemas import EncounterPolicy
from encounter.trust_model import WCLITrustModel
from haic.viability import (
    TTT_BIAS_THRESHOLD,
    TTT_BIAS_WINDOW,
    MAX_TTT_WEIGHT_DRIFT,
    MAX_TTT_UPDATE_COUNT,
    evaluate_ttt_viability,
)


def _make_snapshot(
    update_count: int = 0,
    drift: dict | None = None,
    recent_updates: list | None = None,
) -> dict:
    return {
        "learned_score_weights": {},
        "trust_score_weights": {},
        "drift_from_policy_defaults": drift or {},
        "update_count": update_count,
        "policy_id": "test",
        "recent_updates": recent_updates or [],
    }


def _make_errors(signs: list[int]) -> list[dict]:
    """Build a recent_updates list from a list of +1/-1 sign values."""
    return [{"error": 0.30 * s, "realized_utility": 0.7, "learned_score": 0.7 - 0.30 * s} for s in signs]


# ── weight_drift gate ────────────────────────────────────────────────────────

def test_weight_drift_passes_when_no_drift():
    snap = _make_snapshot(drift={"priority": 0.0, "geometry": 0.0})
    assert evaluate_ttt_viability(snap)["weight_drift"] is True


def test_weight_drift_passes_just_below_threshold():
    snap = _make_snapshot(drift={"priority": MAX_TTT_WEIGHT_DRIFT - 0.001})
    assert evaluate_ttt_viability(snap)["weight_drift"] is True


def test_weight_drift_fails_at_threshold():
    snap = _make_snapshot(drift={"priority": MAX_TTT_WEIGHT_DRIFT + 0.001})
    assert evaluate_ttt_viability(snap)["weight_drift"] is False


def test_weight_drift_fails_on_negative_drift_exceeding_threshold():
    snap = _make_snapshot(drift={"clarity": -(MAX_TTT_WEIGHT_DRIFT + 0.01)})
    assert evaluate_ttt_viability(snap)["weight_drift"] is False


def test_weight_drift_passes_when_drift_is_empty():
    snap = _make_snapshot(drift={})
    assert evaluate_ttt_viability(snap)["weight_drift"] is True


# ── update_rate gate ─────────────────────────────────────────────────────────

def test_update_rate_passes_at_ceiling():
    snap = _make_snapshot(update_count=MAX_TTT_UPDATE_COUNT)
    assert evaluate_ttt_viability(snap)["update_rate"] is True


def test_update_rate_fails_one_above_ceiling():
    snap = _make_snapshot(update_count=MAX_TTT_UPDATE_COUNT + 1)
    assert evaluate_ttt_viability(snap)["update_rate"] is False


def test_update_rate_passes_at_zero():
    snap = _make_snapshot(update_count=0)
    assert evaluate_ttt_viability(snap)["update_rate"] is True


# ── error_bias gate — full-window requirement ────────────────────────────────

def test_error_bias_passes_vacuously_with_fewer_than_full_window():
    """Gate requires exactly TTT_BIAS_WINDOW (10) entries — fewer entries always pass."""
    for n in range(TTT_BIAS_WINDOW):
        errors = _make_errors([+1] * n)  # all same sign, but under window size
        snap = _make_snapshot(recent_updates=errors)
        assert evaluate_ttt_viability(snap)["error_bias"] is True, (
            f"gate should pass vacuously with only {n} entries"
        )


def test_error_bias_can_fire_at_full_window():
    """With exactly TTT_BIAS_WINDOW entries all same sign, gate should fire."""
    errors = _make_errors([+1] * TTT_BIAS_WINDOW)
    snap = _make_snapshot(recent_updates=errors)
    assert evaluate_ttt_viability(snap)["error_bias"] is False


# ── error_bias gate — threshold boundary ────────────────────────────────────

def test_error_bias_passes_at_50_50_split():
    """5 positive + 5 negative = 50% same sign < 70% threshold."""
    errors = _make_errors([+1, -1, +1, -1, +1, -1, +1, -1, +1, -1])
    snap = _make_snapshot(recent_updates=errors)
    assert evaluate_ttt_viability(snap)["error_bias"] is True


def test_error_bias_passes_just_below_threshold():
    """6 of 10 same sign = 60% < 70%."""
    errors = _make_errors([+1, +1, +1, +1, +1, +1, -1, -1, -1, -1])
    snap = _make_snapshot(recent_updates=errors)
    assert evaluate_ttt_viability(snap)["error_bias"] is True


def test_error_bias_fails_at_threshold():
    """7 of 10 same sign = 70% ≥ 70% threshold."""
    errors = _make_errors([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1])
    snap = _make_snapshot(recent_updates=errors)
    assert evaluate_ttt_viability(snap)["error_bias"] is False


def test_error_bias_fails_on_all_negative():
    """All-negative errors should also trigger the gate (negative bias)."""
    errors = _make_errors([-1] * TTT_BIAS_WINDOW)
    snap = _make_snapshot(recent_updates=errors)
    assert evaluate_ttt_viability(snap)["error_bias"] is False


# ── error_bias gate — window advancing (clearing after bias) ─────────────────

def test_error_bias_clears_after_window_diversifies():
    """
    Simulate: 10 biased updates block adaptation, then feed mixed errors via
    record_skipped_observation — gate should clear once the window sees < 70%
    same-sign errors.
    """
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)

    biased_details = {
        "target_priority": 0.95, "geometry_margin": 0.95,
        "duration_margin": 0.95, "imagery_support": 0.95, "clarity_support": 0.95,
    }
    # Seed the log with 7 biased steps via online_update (gate hasn't fired yet)
    for _ in range(7):
        model.online_update(biased_details, learned_score=0.50, realized_utility=0.95)

    # Add 3 more same-sign via record_skipped_observation (simulating post-gate blocking)
    for _ in range(3):
        model.record_skipped_observation(biased_details, learned_score=0.50, realized_utility=0.95)

    snap = model.get_weight_snapshot()
    assert evaluate_ttt_viability(snap)["error_bias"] is False, "gate should fire on 10/10 positive"

    # Now record 4 opposite-sign errors via record_skipped (diversifies window)
    for _ in range(4):
        model.record_skipped_observation(biased_details, learned_score=0.90, realized_utility=0.50)

    snap_after = model.get_weight_snapshot()
    # Window is now 14 entries; last 10 have 6 positive + 4 negative = 60% < 70%
    assert evaluate_ttt_viability(snap_after)["error_bias"] is True, (
        "gate should clear after diversifying errors advance the window"
    )


# ── integration: WCLITrustModel + viability gates ───────────────────────────

def test_weight_drift_fires_after_many_biased_updates():
    """Many same-direction updates should eventually trip weight_drift."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy)

    # Use very high LR and no reg to force drift
    trust_details = {
        "target_priority": 0.0, "geometry_margin": 0.0, "duration_margin": 0.0,
        "imagery_support": 0.0, "clarity_support": 1.0,
    }
    # Push clarity weight up aggressively
    for _ in range(500):
        model.online_update(trust_details, learned_score=0.0, realized_utility=1.0, lr=0.05, reg=0.0)

    snap = model.get_weight_snapshot()
    # At least the clarity weight should have drifted beyond MAX_TTT_WEIGHT_DRIFT
    # (default clarity is 0.12; we pushed it hard with all error signal on clarity)
    # With reg=0, drift can be substantial
    gates = evaluate_ttt_viability(snap)
    assert gates["weight_drift"] is False, "weight_drift should fire after 500 unregularized biased updates"
