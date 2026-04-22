"""
SimSat → MuZero AbstractGame adapter.

Mirrors the structure of `arc3_game.py` (ARC-AGI-3 port) but targets the
SimSat encounter-planning loop. Wraps `SimSatEnv` so MuZero self-play,
training, and TTT can run against either the offline-replay corpus or a
live SimSat backend without changes to the muzero-general internals.

Action encoding
---------------
Four discrete trust actions — accept / defer / skip / refine. See
`simsat_env.TRUST_ACCEPT` / `TRUST_DEFER` / `TRUST_SKIP` / `TRUST_REFINE`.

The extended-action variant (target selection) is left for later; bump
`action_space` when wiring it.

Observation
-----------
(1, 64, 64) float32 in [0, 1]. Source is a normalized Sentinel tile from
`SimSatEnv._tile_to_obs()`. Nearest-neighbour resize preserves band-value
identity — analogous to ARC3's colour-palette preservation.

Reward shaping
--------------
Delegated to `SimSatEnv._compute_reward()`:
    -0.01  per step
    +util  when action=accept and useful materialization
    -0.10  when action=accept and not-useful materialization
     0.00  when action=defer
    -0.01  when action=skip and tile was actually useful (missed opportunity)
    -0.05  when action=refine (secondary evidence pass cost)

Window tracking
---------------
One encounter window per env.step() — matches ARC3's per-level boundary
semantics. `game.current_window` and `game.window_history` are the
Scope-2 TTT inputs.

TTT scopes
----------
Same structure as arc3_game.py — TTTScope1 (intra-window) and TTTScope2
(cross-window). Default budgets may need tuning: SimSat scenario evals
can have more windows than ARC3 games have levels, so the default decay
(0.65) may floor to 1 step faster than desired. See TTTScope.ttt_budget
for the schedule.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import sys
from typing import Any, Optional

import numpy as np

from simsat_env import (
    OBS_H,
    OBS_W,
    SimSatEnv,
    TRUST_ACCEPT,
    TRUST_DEFER,
    TRUST_REFINE,
    TRUST_SKIP,
    TRUST_ACTION_NAMES,
    EncounterRecord,
)

_log = logging.getLogger(__name__)


# Allow running from neurosym/ or sim/ root (same pattern as arc3_game.py)
_HERE = pathlib.Path(__file__).resolve().parent


# muzero-general AbstractGame — imported lazily so this file works without
# the muzero deps installed (e.g. for inspection or type-checking).
try:
    _mz_root = _HERE.parent / "external" / "muzero-general"
    if str(_mz_root) not in sys.path:
        sys.path.insert(0, str(_mz_root))
    from games.abstract_game import AbstractGame as _AbstractGame
except ImportError:
    class _AbstractGame:  # type: ignore
        """Stub fallback so this module is importable without muzero-general."""
        pass


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

class MuZeroConfig:
    """
    MuZero hyperparameters for SimSat.

    Defaults ported from arc3_game.py's MuZeroConfig; tune as needed.
    Starter action space is the 4-vocab trust set; extend by overriding
    `action_space` and `action_space_size` after construction for per-target
    selection.
    """

    def __init__(self, game_id: str = "simsat", scenario_pack: str = "all"):
        # ── Game ──────────────────────────────────────────────────────
        self.game_id = game_id
        self.scenario_pack = scenario_pack

        # Frame is (1, 64, 64); normalised to float32 [0, 1]
        self.observation_shape = (1, OBS_H, OBS_W)

        # Trust actions only (starter). Extend for target selection.
        self.action_space = list(range(4))  # accept / defer / skip / refine

        self.players = list(range(1))
        self.stacked_observations = 0

        self.muzero_player = 0
        self.opponent = None

        # ── Self-play ──────────────────────────────────────────────────────
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 80                  # Matches ARC3 budget; tune for SimSat scenarios
        self.num_simulations = 25            # Keep low for TTT speed
        self.discount = 0.99
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # ── Network ──────────────────────────────────────────────────────────
        # IMPALA CNN encoder: no BatchNorm → stable TTT under small batches.
        # Hidden state: (B, 256, 1, 1) — spatial dims collapsed to 1×1.
        self.network = "impala"
        self.support_size = 10

        # IMPALA-specific parameters
        self.impala_channels = (16, 32, 32)  # channel sizes per stack
        self.impala_out_dim  = 256           # encoder output / hidden-state width

        # Dynamics / prediction head residual blocks (operate on 1×1 spatial)
        self.blocks   = 1
        self.channels = 256                  # must equal impala_out_dim

        # Head reduction
        self.reduced_channels_reward = 32
        self.reduced_channels_value  = 32
        self.reduced_channels_policy = 32
        self.resnet_fc_reward_layers = [64]
        self.resnet_fc_value_layers  = [64]
        self.resnet_fc_policy_layers = [64]

        # ResNet fallback (unused when network="impala")
        self.downsample = "CNN"

        self.encoding_size = 64              # for fullyconnected fallback (unused)
        self.fc_representation_layers = [64]
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]

        # ── Training ─────────────────────────────────────────────────────────
        # torch imported lazily so this config is inspectable without the full
        # MuZero training environment installed (matches arc3_game.py's
        # arcengine pattern). `train_on_gpu` defaults to False when torch is
        # missing; set manually when driving the actual Trainer.
        try:
            import torch  # noqa: F401
            _cuda_ok = torch.cuda.is_available()
        except ImportError:
            _cuda_ok = False
        self.results_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / f"simsat_{scenario_pack}"
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )
        self.save_model = True
        self.training_steps = 20_000
        self.batch_size = 64
        self.checkpoint_interval = 50
        self.value_loss_weight = 0.25
        self.train_on_gpu = _cuda_ok

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 3e-4
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 5_000

        # ── Replay buffer ───────────────────────────────────────────────────
        self.replay_buffer_size = 1_000
        self.num_unroll_steps = 5
        self.td_steps = 20
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5

        # ── Required by muzero-general internals ────────────────────────────────
        self.seed = 0

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """Match arc3_game.py's temperature schedule."""
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        return 0.25


# --------------------------------------------------------------------------- #
# Game adapter                                                                #
# --------------------------------------------------------------------------- #

class SimSatGame(_AbstractGame):
    """
    MuZero AbstractGame adapter for a SimSat encounter-planning episode.

    Parameters
    ----------
    records : list[EncounterRecord] | None
        Encounter records for offline replay backend. Required when
        `backend="offline_replay"`.
    backend : str
        "offline_replay" or "api" — passed through to SimSatEnv.
    scenario_pack : str
        Filter applied when using offline replay.
    api_base_url : str | None
        Only used when backend="api".
    step_penalty : float
        Passed to SimSatEnv.
    seed : int | None
        Unused — SimSatEnv is deterministic. Kept for AbstractGame compat.
    """

    def __init__(
        self,
        records: Optional[list[EncounterRecord]] = None,
        backend: str = "offline_replay",
        scenario_pack: str = "all",
        api_base_url: Optional[str] = None,
        step_penalty: float = -0.01,
        seed: Optional[int] = None,
    ):
        self.scenario_pack = scenario_pack
        self._env = SimSatEnv(
            backend=backend,
            records=records,
            scenario_pack=scenario_pack,
            api_base_url=api_base_url,
            step_penalty=step_penalty,
            seed=seed,
        )
        # Level-tracking mirror for TTT scope access. `current_window` plays
        # the role of `current_level` in arc3_game.py.
        self.current_window: int = 0
        self.window_history: list[dict] = []
        self._step_count: int = 0

    # ── AbstractGame interface ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        self.current_window = self._env.current_window
        self.window_history = []
        self._step_count = 0
        return obs

    def step(self, action_int: int):
        obs, reward, done, info = self._env.step(action_int)
        self._step_count += 1
        # Mirror the env's window_history so TTTScope can read it off the game.
        self.window_history = list(self._env.window_history)
        self.current_window = self._env.current_window
        return obs, reward, done

    def legal_actions(self) -> list[int]:
        return self._env.legal_actions()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def action_to_string(self, action_int: int) -> str:
        return self._env.action_to_string(action_int)


# --------------------------------------------------------------------------- #
# TTT scopes — ported directly from arc3_game.py                              #
# --------------------------------------------------------------------------- #

class TTTScope:
    """
    Base class for test-time training wrappers around SimSatGame.

    Tracks prediction errors per encounter window and adjusts the gradient-step
    budget via surprise-driven decay: fewer steps when the model's prediction
    loss is already low (accumulated world model fits the current context).

    Defaults match arc3_game.py. Tune if SimSat scenario evals go deeper than
    ARC3 games' ~7-level depth — the 0.65 decay factor can floor to 1 step
    after ~8 windows and leave later windows without meaningful TTT.
    """

    # TTT budget at window 1 (max). Decays each window.
    BASE_TTT_STEPS = 32
    DECAY_FACTOR = 0.65  # per-window exponential decay

    def __init__(self, game: SimSatGame):
        self.game = game
        self._window_ttt_steps: dict[int, int] = {}

    def ttt_budget(self, window: int) -> int:
        """
        Gradient steps to allocate for TTT at the start of `window`.

        Window 1 gets BASE_TTT_STEPS. Each subsequent window gets
        DECAY_FACTOR × prior window's budget (floored at 1).

        The decay is surprise-driven in the full implementation; here it is
        a fixed exponential schedule as a conservative baseline. Replace with
        observed-loss-driven adjustment once training loop is wired.
        """
        steps = max(1, int(self.BASE_TTT_STEPS * (self.DECAY_FACTOR ** (window - 1))))
        self._window_ttt_steps[window] = steps
        return steps

    def window_transitions(self) -> list[dict]:
        """Return recorded window boundary events from the underlying game."""
        return self.game.window_history


class TTTScope1(TTTScope):
    """
    Intra-window adaptation.

    On each new encounter window, performs `ttt_budget(window)` gradient steps
    on the most recent (obs, action, reward, next_obs) tuples from that window
    to tune the dynamics/prediction heads to this window's specific context
    (target, cloud, geometry, etc.).
    """
    pass  # gradient logic lives in the MuZero trainer; this provides budget


class TTTScope2(TTTScope):
    """
    Cross-window accumulation.

    Carries the model state learned from windows 1..W-1 into window W.
    Selectively re-weights prior window data based on feature similarity
    (windows with similar contexts contribute more to the prior).

    The accumulated knowledge makes TTT at window W cheaper — budget decays
    naturally because the model's prediction error at window start is already
    lower when the world model is well-calibrated.
    """

    # Slightly stronger retention of prior windows vs. pure intra-window TTT
    BASE_TTT_STEPS = 16
    DECAY_FACTOR = 0.70


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-pack", default="all")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    # Fabricate a tiny synthetic corpus so we can smoke-test without the
    # real 86-trace store.
    def _fake_records(n: int) -> list[EncounterRecord]:
        rng = np.random.default_rng(0)
        packs = ["maritime_chokepoints", "disaster_response_weather", "urban_coastal_ambiguity"]
        targets = ["suez_canal", "houston_ship", "sf_bay"]
        records = []
        for i in range(n):
            records.append(EncounterRecord(
                window_id=f"smoke_{i}",
                target_id=targets[i % len(targets)],
                scenario_pack=packs[i % len(packs)],
                tile=rng.integers(0, 2500, size=(1, 64, 64), dtype=np.int16).astype(np.float32),
                target_priority="high_priority",
                sentinel_available=True,
                sentinel_cloud_cover=float(rng.uniform(0, 30)),
                elevation_degrees=float(rng.uniform(30, 80)),
                line_of_sight=True,
                useful=bool(i % 2 == 0),
                utility_realized=float(rng.uniform(0.7, 0.95)),
            ))
        return records

    records = _fake_records(6)
    game = SimSatGame(records=records, scenario_pack=args.scenario_pack)
    obs = game.reset()
    print(
        f"Smoke-testing SimSatGame(scenario_pack='{args.scenario_pack}')"
    )
    print(
        f"  reset OK — obs shape={obs.shape}, dtype={obs.dtype}, "
        f"range=[{obs.min():.3f}, {obs.max():.3f}]"
    )
    print(f"  legal_actions={game.legal_actions()}")

    scope1 = TTTScope1(game)
    for step in range(args.steps):
        action = game.legal_actions()[step % len(game.legal_actions())]
        obs, reward, done = game.step(action)
        budget = scope1.ttt_budget(game.current_window)
        print(
            f"  step {step+1}: action={game.action_to_string(action)}, "
            f"reward={reward:+.3f}, done={done}, window={game.current_window}, "
            f"ttt_budget={budget}"
        )
        if done:
            break

    print("Smoke test complete.")
