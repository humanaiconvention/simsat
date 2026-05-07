"""Edge-case + stress tests for WCLITrustModel and TTT viability gates.

Covers numerical extremes, bias-window boundary conditions, log-cap behavior,
and recovery from extended biased operation.
"""
from __future__ import annotations

from encounter.schemas import EncounterPolicy
from encounter.trust_model import WCLITrustModel, _MAX_UPDATE_LOG
from haic.viability import (
    TTT_BIAS_WINDOW,
    evaluate_ttt_viability,
)


# ── Numerical edge cases ────────────────────────────────────────────────────

def test_weights_remain_normalized_after_extreme_realized_utility():
    """Weights must sum to 1 even with realized_utility at the extremes."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.99, "geometry_margin": 0.99,
        "duration_margin": 0.99, "imagery_support": 0.99, "clarity_support": 0.99,
    }

    # realized_utility = 0.0 (operator says "useless")
    model.online_update(details, learned_score=0.99, realized_utility=0.0)
    assert abs(sum(model._learned_weights.values()) - 1.0) < 1e-6
    assert all(w >= 0.001 for w in model._learned_weights.values())

    # realized_utility = 1.0 (operator says "maximally useful")
    model.online_update(details, learned_score=0.0, realized_utility=1.0)
    assert abs(sum(model._learned_weights.values()) - 1.0) < 1e-6
    assert all(w >= 0.001 for w in model._learned_weights.values())


def test_weights_floor_holds_under_repeated_negative_pressure():
    """Repeatedly pushing a weight toward zero hits the 0.001 floor and holds."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    # Push priority weight down: realized < learned, with priority feature high
    details = {
        "target_priority": 0.99,  # only priority active
        "geometry_margin": 0.0, "duration_margin": 0.0,
        "imagery_support": 0.0, "clarity_support": 0.0,
    }
    for _ in range(2000):
        model.online_update(details, learned_score=0.99, realized_utility=0.0,
                            lr=0.5, reg=0.0)
    # priority weight should hit floor; all weights still >= 0.001
    assert all(w >= 0.001 for w in model._learned_weights.values())
    assert abs(sum(model._learned_weights.values()) - 1.0) < 1e-6


def test_zero_error_step_is_idempotent_modulo_reg():
    """When realized == learned, only the L2 reg term moves weights."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    defaults = dict(policy.learned_score_weights)
    # Start with weights mutated away from defaults
    model._learned_weights = {k: defaults[k] * 1.5 for k in defaults}
    # Renormalize so they still sum to 1
    total = sum(model._learned_weights.values())
    model._learned_weights = {k: v / total for k, v in model._learned_weights.items()}

    weights_before = dict(model._learned_weights)
    details = {
        "target_priority": 0.5, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.5,
    }
    model.online_update(details, learned_score=0.5, realized_utility=0.5)

    # Each weight should have moved toward the default by ~ reg * (default - current)
    for k in defaults:
        delta = model._learned_weights[k] - weights_before[k]
        expected_dir = defaults[k] - weights_before[k]
        # The reg pull moves weight in same direction as (default - current)
        # Renormalization may slightly perturb the magnitude — check direction only.
        if abs(expected_dir) > 0.001:
            assert delta * expected_dir >= 0, f"weight {k} moved against L2 reg direction"


# ── Bias window boundary conditions ─────────────────────────────────────────

def test_bias_window_with_exactly_zero_errors_passes():
    """Errors of exactly 0.0 are neither positive nor negative — gate passes."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.5, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.5,
    }
    # 10 perfectly zero-error updates
    for _ in range(TTT_BIAS_WINDOW):
        model.online_update(details, learned_score=0.5, realized_utility=0.5)

    snap = model.get_weight_snapshot()
    gates = evaluate_ttt_viability(snap)
    # With zero errors, neither sign dominates; gate must pass
    assert gates["error_bias"] is True


def test_bias_window_with_mixed_positive_zero_errors():
    """6 positive + 4 zero errors: positive count = 6/10 = 60% < 70% threshold."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.5, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.5,
    }
    # 6 positive errors, 4 zero errors
    for _ in range(6):
        model.online_update(details, learned_score=0.4, realized_utility=0.7)
    for _ in range(4):
        model.online_update(details, learned_score=0.5, realized_utility=0.5)

    snap = model.get_weight_snapshot()
    gates = evaluate_ttt_viability(snap)
    # 6/10 = 60% same-sign < 70% threshold
    assert gates["error_bias"] is True


# ── Log-cap stress test ─────────────────────────────────────────────────────

def test_update_log_caps_at_max_update_log():
    """The internal _update_log is capped at _MAX_UPDATE_LOG entries."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.5, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.5,
    }
    # Run more updates than the cap
    n_runs = _MAX_UPDATE_LOG + 100
    for _ in range(n_runs):
        model.online_update(details, learned_score=0.5, realized_utility=0.5)

    assert len(model._update_log) == _MAX_UPDATE_LOG
    # update_count tracks all updates, not capped
    assert model.update_count == n_runs


def test_record_skipped_observation_caps_log_too():
    """record_skipped_observation also respects the log cap."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.5, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.5,
    }
    n_runs = _MAX_UPDATE_LOG + 50
    for _ in range(n_runs):
        model.record_skipped_observation(details, learned_score=0.5, realized_utility=0.5)

    assert len(model._update_log) == _MAX_UPDATE_LOG


def test_bias_window_correct_after_log_overflow():
    """After log overflow, the bias window still reads the most recent 10 entries."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.5, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.5,
    }
    # Fill the log past its cap with neutral errors
    for _ in range(_MAX_UPDATE_LOG + 50):
        model.online_update(details, learned_score=0.5, realized_utility=0.5)

    # Now do 10 biased updates — gate should fire
    for _ in range(TTT_BIAS_WINDOW):
        model.online_update(details, learned_score=0.3, realized_utility=0.9)

    snap = model.get_weight_snapshot()
    gates = evaluate_ttt_viability(snap)
    # Last 10 entries are all positive errors
    assert gates["error_bias"] is False


# ── Long-run recovery ───────────────────────────────────────────────────────

def test_recovery_from_extended_bias_via_record_skipped():
    """
    Simulate prolonged biased operator feedback (bias gate keeps blocking),
    then a switch to balanced feedback. The gate must clear cleanly.
    """
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    details = {
        "target_priority": 0.7, "geometry_margin": 0.7,
        "duration_margin": 0.7, "imagery_support": 0.7, "clarity_support": 0.7,
    }
    # 50 biased updates via online_update first to fully populate window
    for _ in range(50):
        model.online_update(details, learned_score=0.4, realized_utility=0.95)

    snap = model.get_weight_snapshot()
    assert evaluate_ttt_viability(snap)["error_bias"] is False

    # Now 10 more bias-direction observations are blocked (record_skipped)
    for _ in range(10):
        model.record_skipped_observation(details, learned_score=0.4, realized_utility=0.95)

    # Still firing
    snap = model.get_weight_snapshot()
    assert evaluate_ttt_viability(snap)["error_bias"] is False

    # Now switch to opposite-direction feedback (10 negative errors via record_skipped)
    for _ in range(10):
        model.record_skipped_observation(details, learned_score=0.95, realized_utility=0.40)

    # Window now contains 10 negative errors (last 10) — same-sign 100% (negative).
    # That's still ≥ 70% same-sign. Gate stays fired.
    snap = model.get_weight_snapshot()
    assert evaluate_ttt_viability(snap)["error_bias"] is False

    # Mix it: 6 negative + 4 positive ⇒ 60% same-sign ⇒ gate clears
    for _ in range(4):
        model.record_skipped_observation(details, learned_score=0.4, realized_utility=0.9)

    snap = model.get_weight_snapshot()
    assert evaluate_ttt_viability(snap)["error_bias"] is True


# ── Drift bound under regularization ────────────────────────────────────────

def test_l2_reg_bounds_drift_under_realistic_mixed_feedback():
    """With reg=0.002 and realistic mixed feedback, drift stays under the weight_drift gate."""
    import random
    rng = random.Random(42)
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    # Realistic: features vary, errors are roughly balanced across the corpus
    for _ in range(500):
        details = {
            "target_priority": rng.uniform(0.3, 0.95),
            "geometry_margin": rng.uniform(0.3, 0.9),
            "duration_margin": rng.uniform(0.3, 0.9),
            "imagery_support": rng.uniform(0.3, 1.0),
            "clarity_support": rng.uniform(0.0, 1.0),
        }
        learned = (
            0.20 * details["target_priority"] + 0.34 * details["geometry_margin"]
            + 0.18 * details["duration_margin"] + 0.16 * details["imagery_support"]
            + 0.12 * details["clarity_support"]
        )
        realized = max(0.0, min(1.0, learned + rng.gauss(0, 0.10)))
        model.online_update(details, learned_score=learned, realized_utility=realized)

    snap = model.get_weight_snapshot()
    gates = evaluate_ttt_viability(snap)
    # Realistic feedback with reg=0.002 keeps drift bounded — weight_drift must NOT fire
    assert gates["weight_drift"] is True, (
        f"weight_drift gate fired under realistic feedback — drift={snap['drift_from_policy_defaults']}"
    )


def test_weight_drift_gate_fires_under_adversarial_single_feature_pressure():
    """Adversarial setup: all error signal on one feature for 500 cycles → weight_drift correctly fires.

    This is the safety story: reg alone can't bound drift under sustained adversarial input.
    The weight_drift gate is the second line of defense."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    # Adversarial: every step rewards clarity exclusively, error is strongly positive
    details = {
        "target_priority": 0.0, "geometry_margin": 0.0,
        "duration_margin": 0.0, "imagery_support": 0.0, "clarity_support": 1.0,
    }
    for _ in range(500):
        model.online_update(details, learned_score=0.0, realized_utility=1.0)

    snap = model.get_weight_snapshot()
    gates = evaluate_ttt_viability(snap)
    # The gate MUST fire — this is the architectural claim about safety rails
    assert gates["weight_drift"] is False, "weight_drift gate failed to detect adversarial single-feature drift"


def test_get_weight_snapshot_returns_consistent_drift_after_update():
    """drift_from_policy_defaults must equal weights - defaults at every snapshot."""
    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    defaults = dict(policy.learned_score_weights)
    details = {
        "target_priority": 0.8, "geometry_margin": 0.5,
        "duration_margin": 0.5, "imagery_support": 0.5, "clarity_support": 0.7,
    }
    for _ in range(20):
        model.online_update(details, learned_score=0.5, realized_utility=0.85)

    snap = model.get_weight_snapshot()
    weights = snap["learned_score_weights"]
    drift = snap["drift_from_policy_defaults"]
    for k in defaults:
        # Allow small rounding tolerance — both values are rounded to 6 decimals
        expected = round(weights[k] - defaults[k], 6)
        assert abs(drift[k] - expected) < 1e-6


# ── Property test: weight snapshot internal consistency ─────────────────────

def test_weight_snapshot_internally_consistent_over_random_operator_labels():
    """Feed 100 random operator labels through online_update + record_skipped_observation
    and assert the resulting weight snapshot is internally consistent:
      - weights sum to 1.0
      - all weights are positive
      - drift_from_policy_defaults[k] == learned_score_weights[k] - default[k]
      - update_count + skipped entries match log lengths
    """
    import random
    rng = random.Random(42)

    policy = EncounterPolicy()
    model = WCLITrustModel(policy=policy)
    defaults = dict(policy.learned_score_weights)

    actions = ["accept", "refine", "defer", "skip"]
    details_pool = [
        {"target_priority": rng.uniform(0, 1), "geometry_margin": rng.uniform(0, 1),
         "duration_margin": rng.uniform(0, 1), "imagery_support": rng.uniform(0, 1),
         "clarity_support": rng.uniform(0, 1)}
        for _ in range(100)
    ]

    expected_updates = 0
    expected_skips = 0

    for i in range(100):
        details = details_pool[i]
        learned = rng.uniform(0.0, 1.0)
        realized = rng.uniform(0.0, 1.0)
        if rng.random() < 0.7:
            model.online_update(details, learned_score=learned, realized_utility=realized)
            expected_updates += 1
        else:
            model.record_skipped_observation(details, learned_score=learned, realized_utility=realized)
            expected_skips += 1

    snap = model.get_weight_snapshot()
    weights = snap["learned_score_weights"]
    drift = snap["drift_from_policy_defaults"]

    # Weights sum to 1
    assert abs(sum(weights.values()) - 1.0) < 1e-5, (
        f"weights don't sum to 1: {sum(weights.values())}"
    )
    # All weights positive
    assert all(v > 0 for v in weights.values()), (
        f"negative weight found: {weights}"
    )
    # Drift matches weights - defaults
    for k in defaults:
        expected_drift = round(weights[k] - defaults[k], 6)
        assert abs(drift[k] - expected_drift) < 1e-5, (
            f"drift mismatch for {k}: snap says {drift[k]}, computed {expected_drift}"
        )
    # update_count matches
    assert model.update_count == expected_updates
    # Log length is capped at _MAX_UPDATE_LOG, total entries from both paths
    total_logged = expected_updates + expected_skips
    assert len(model._update_log) == min(total_logged, _MAX_UPDATE_LOG)
