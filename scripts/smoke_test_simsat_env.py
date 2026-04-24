"""Smoke test for SimSatEnv and SimSatGame.

Runs without any external dependencies beyond numpy — no Django, no torch,
no muzero-general clone. Generates a synthetic 6-record encounter corpus and
steps through it, verifying:

  - reset produces a (1, 64, 64) observation in [0, 1]
  - step with each of the four trust actions returns (obs, reward, done, info)
  - reward shaping fires correctly for accept+useful, skip+missed, refine
  - window tracking increments correctly
  - TTTScope1/TTTScope2 budgets decay as expected

Usage:
    python scripts/smoke_test_simsat_env.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/sim/muzero importable regardless of CWD
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from sim.muzero.simsat_env import (
    OBS_H, OBS_W,
    EncounterRecord, SimSatEnv,
    TRUST_ACCEPT, TRUST_DEFER, TRUST_SKIP, TRUST_REFINE,
)


def _make_records() -> list[EncounterRecord]:
    """Six synthetic encounter records spanning all three scenario packs."""
    rng = np.random.default_rng(42)
    packs = ["maritime_chokepoints", "disaster_response_weather", "urban_coastal_ambiguity"]
    targets = ["suez_canal", "houston_ship", "sf_bay"]
    records = []
    for i in range(6):
        records.append(EncounterRecord(
            window_id=f"smoke_{i:02d}",
            target_id=targets[i % 3],
            scenario_pack=packs[i % 3],
            tile=rng.integers(0, 2500, size=(1, 64, 64), dtype=np.int16).astype(np.float32),
            target_priority="high_priority",
            sentinel_available=True,
            sentinel_cloud_cover=float(rng.uniform(0, 25)),
            elevation_degrees=float(rng.uniform(30, 80)),
            line_of_sight=True,
            useful=(i % 2 == 0),
            utility_realized=float(rng.uniform(0.7, 0.95)),
        ))
    return records


def test_env_reset_and_step() -> None:
    records = _make_records()
    env = SimSatEnv(records=records)
    obs = env.reset()
    assert obs.shape == (1, OBS_H, OBS_W), f"expected (1, {OBS_H}, {OBS_W}) got {obs.shape}"
    assert obs.dtype == np.float32, f"expected float32 got {obs.dtype}"
    assert 0.0 <= obs.min() and obs.max() <= 1.0, f"obs range [{obs.min()}, {obs.max()}] out of [0, 1]"
    print(f"  ✓ reset → obs {obs.shape} {obs.dtype} range [{obs.min():.3f}, {obs.max():.3f}]")

    # Step through all 6 records with cycled actions
    actions = [TRUST_ACCEPT, TRUST_DEFER, TRUST_SKIP, TRUST_REFINE, TRUST_ACCEPT, TRUST_SKIP]
    done_at = None
    total_reward = 0.0
    for i, a in enumerate(actions):
        next_obs, reward, done, info = env.step(a)
        total_reward += reward
        assert next_obs.shape == (1, OBS_H, OBS_W)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info["action_name"] in ("accept", "defer", "skip", "refine")
        if done:
            done_at = i
            break
    assert done_at == 5, f"expected done at step 5 (last of 6), got {done_at}"
    print(f"  ✓ stepped through 6 records, total_reward={total_reward:+.3f}, done at step 5")


def test_reward_shaping() -> None:
    """Verify reward responds to useful vs not-useful and to refine cost."""
    records = _make_records()

    # Force a known-useful record into position 0 so accept there is rewarded
    records[0].useful = True
    records[0].utility_realized = 0.95

    env = SimSatEnv(records=records)
    env.reset()
    _, reward, _, _ = env.step(TRUST_ACCEPT)
    # step_penalty (-0.01) + utility_realized (0.95) = ~0.94
    assert 0.90 < reward < 0.96, f"accept+useful reward {reward} out of expected ~0.94 band"
    print(f"  ✓ accept+useful: reward={reward:+.3f} (~0.94 expected)")

    # Force a known-not-useful record at position 1
    env2 = SimSatEnv(records=[records[0], records[1], records[2]])
    # Rewrite position 1 explicitly: not useful
    env2._records[1].useful = False
    env2._records[1].utility_realized = None
    env2.reset()
    env2.step(TRUST_DEFER)  # skip past position 0 neutrally
    _, reward, _, _ = env2.step(TRUST_ACCEPT)
    # step_penalty (-0.01) + materialize_not_useful_penalty (-0.10) = -0.11
    assert -0.13 < reward < -0.09, f"accept+not-useful reward {reward} out of expected -0.11 band"
    print(f"  ✓ accept+not_useful: reward={reward:+.3f} (~-0.11 expected)")

    # Refine cost
    env3 = SimSatEnv(records=_make_records())
    env3.reset()
    _, reward, _, _ = env3.step(TRUST_REFINE)
    # step_penalty (-0.01) + refine_cost (-0.05) = -0.06
    assert -0.08 < reward < -0.04, f"refine reward {reward} out of expected -0.06 band"
    print(f"  ✓ refine: reward={reward:+.3f} (~-0.06 expected)")


def test_legal_actions_and_invalid() -> None:
    env = SimSatEnv(records=_make_records())
    env.reset()
    legal = env.legal_actions()
    assert legal == [0, 1, 2, 3], f"expected [0,1,2,3] got {legal}"

    # Invalid action should raise
    try:
        env.step(99)
        raise AssertionError("expected ValueError for invalid action")
    except ValueError:
        pass
    print(f"  ✓ legal_actions={legal}, invalid action correctly rejected")


def test_scenario_pack_filter() -> None:
    records = _make_records()
    maritime_only = SimSatEnv(
        records=records,
        scenario_pack="maritime_chokepoints",
    )
    maritime_only.reset()
    # Out of 6 records, 2 are maritime (indices 0, 3)
    count = 0
    while True:
        _, _, done, info = maritime_only.step(TRUST_DEFER)
        count += 1
        if done:
            break
    assert count == 2, f"expected 2 maritime records, got {count}"
    print(f"  ✓ scenario_pack filter: {count} records matched 'maritime_chokepoints'")


def test_simsat_game_and_ttt_scopes() -> None:
    from sim.muzero.simsat_game import MuZeroConfig, SimSatGame, TTTScope1, TTTScope2

    # Config smoke
    cfg = MuZeroConfig(game_id="simsat", scenario_pack="all")
    assert cfg.observation_shape == (1, OBS_H, OBS_W)
    assert cfg.action_space == [0, 1, 2, 3]
    assert cfg.network == "impala"
    print(f"  ✓ MuZeroConfig: obs={cfg.observation_shape} action_space={cfg.action_space} network={cfg.network}")

    # Game wrapper
    records = _make_records()
    game = SimSatGame(records=records)
    obs = game.reset()
    assert obs.shape == (1, OBS_H, OBS_W)
    assert game.legal_actions() == [0, 1, 2, 3]
    print(f"  ✓ SimSatGame: reset → obs {obs.shape}, legal_actions={game.legal_actions()}")

    # TTT scopes with budget decay
    scope1 = TTTScope1(game)
    scope2 = TTTScope2(game)
    # Step through the first 4 windows and record budgets
    budgets1 = []
    budgets2 = []
    for step in range(4):
        budgets1.append(scope1.ttt_budget(game.current_window))
        budgets2.append(scope2.ttt_budget(game.current_window))
        game.step(TRUST_DEFER)
    # Expect monotonic decrease for both scopes
    for i in range(len(budgets1) - 1):
        assert budgets1[i] >= budgets1[i + 1], f"scope1 budget should decrease: {budgets1}"
        assert budgets2[i] >= budgets2[i + 1], f"scope2 budget should decrease: {budgets2}"
    assert budgets1[0] == 32, f"scope1 base should be 32, got {budgets1[0]}"
    assert budgets2[0] == 16, f"scope2 base should be 16, got {budgets2[0]}"
    print(f"  ✓ TTTScope1 budgets: {budgets1} (base 32, decay 0.65)")
    print(f"  ✓ TTTScope2 budgets: {budgets2} (base 16, decay 0.70)")

    # Window history accumulates
    assert len(game.window_history) == 4
    print(f"  ✓ window_history has {len(game.window_history)} entries as expected")


def main() -> int:
    print("=" * 70)
    print("Phase 5 T3 smoke tests — simsat_env + simsat_game")
    print("=" * 70)

    print("\n[1] Env reset and step")
    test_env_reset_and_step()

    print("\n[2] Reward shaping")
    test_reward_shaping()

    print("\n[3] Legal actions and invalid rejection")
    test_legal_actions_and_invalid()

    print("\n[4] Scenario-pack filter")
    test_scenario_pack_filter()

    print("\n[5] SimSatGame + TTT scopes")
    test_simsat_game_and_ttt_scopes()

    print("\n" + "=" * 70)
    print("All smoke tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
