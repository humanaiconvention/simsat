"""
SimSat environment — thin Gym-style wrapper over the encounter-planning loop.

Serves the same role for SimSat that `arcade.Arcade` + `env.step(action)` serves
for the ARC-AGI-3 MuZero pipeline: produce (observation, reward, done) tuples
from discrete actions so MuZero / TTT / MCTS machinery can run unchanged.

Two backends are supported:

  * `offline_replay` (default) — plays through a pre-stored sequence of
    encounter-window records (Sentinel tile + metadata + decision + outcome).
    Used for pretraining and stage-2 fine-tuning where we need deterministic,
    reproducible rollouts from the 86-trace corpus. No Django backend required.

  * `api` — drives a running SimSat Django backend at a configured base URL.
    Used for stage-3 TTT at deployment time. Analog of ARC3Game talking to
    `three.arcprize.org`. Stubbed here; wire the actual HTTP calls against
    `/sim/encounter/*` endpoints once ready.

ACTION SPACE
------------
Four discrete trust actions, matching WCLITrustModel's vocabulary:

    0 → accept    materialize now, high confidence
    1 → defer     wait for a better window
    2 → skip      do not materialize
    3 → refine    pay for a cheaper evidence pass first

An optional extended action space adds per-target selection, doubling
|actions| for each active target. Default keeps it at 4 for starter simplicity.

OBSERVATION
-----------
Raw Sentinel tile resized to (1, 64, 64) float32 normalised to [0, 1] by
default (matches ARC3 for clean port). Resize uses nearest-neighbour to
preserve band-value identity — analogous to ARC3's colour-palette preservation.

For multispectral later: accept (C, 64, 64) with C > 1.

REWARD SHAPING
--------------
    -0.01   step penalty (efficiency nudge)
    +util   if action=accept and observation materialized useful (util in [0,1])
    -0.10   if action=accept and observation materialized not useful (wasted
            materialization cost)
     0.00   if action=defer (no reward, no cost)
    -0.01   if action=skip when target was actually useful (missed opportunity)
    -0.05   if action=refine (cost of secondary evidence pass)

Tune via constructor kwargs. These defaults are placeholders; real reward
calibration happens during stage-1 pretraining.

ENCOUNTER WINDOW (LEVEL) TRACKING
---------------------------------
`self.current_window` — 1-indexed; increments each step().
`self.window_history` — [{step, window, reward, useful}] for TTT Scope 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

TRUST_ACCEPT: int = 0
TRUST_DEFER:  int = 1
TRUST_SKIP:   int = 2
TRUST_REFINE: int = 3

TRUST_ACTION_NAMES = {
    TRUST_ACCEPT: "accept",
    TRUST_DEFER:  "defer",
    TRUST_SKIP:   "skip",
    TRUST_REFINE: "refine",
}

# Sentinel normalization constant. Sentinel-2 L2A reflectance is typically
# scaled by 10_000 to int16; clamp + divide gives [0, 1]. Tune per band if
# using multispectral.
SENTINEL_NORM: float = 10_000.0

# Observation grid size — matches ARC3 for structural reuse.
OBS_H: int = 64
OBS_W: int = 64


# --------------------------------------------------------------------------- #
# Encounter record (offline replay format)                                    #
# --------------------------------------------------------------------------- #

@dataclass
class EncounterRecord:
    """Single encounter-window record for offline replay.

    This is the tuple the existing 86-trace ObservationVLA corpus can be
    rehydrated into. `trace_to_inputs()` in the parity harness (Phase 5 T1)
    already extracts most of these fields from stored JSON; a short adapter
    converts those into EncounterRecord.
    """

    window_id: str
    target_id: str
    scenario_pack: str

    # Raw Sentinel tile — shape (H, W) or (C, H, W). Env will normalize.
    tile: np.ndarray

    # Target metadata
    target_priority: str = "normal"
    target_tags: list[str] = field(default_factory=list)

    # Probe metadata (Sentinel availability, cloud, geometry)
    sentinel_available: Optional[bool] = None
    sentinel_cloud_cover: Optional[float] = None   # 0..100
    elevation_degrees: Optional[float] = None
    line_of_sight: Optional[bool] = None

    # Optional ground-truth outcome (if stage-2 training from labelled traces)
    actual_action: Optional[str] = None            # "accept" | "defer" | "skip" | "refine"
    utility_realized: Optional[float] = None       # 0..1 when materialized
    useful: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Env                                                                         #
# --------------------------------------------------------------------------- #

class SimSatEnv:
    """Gym-style env for the SimSat encounter planning loop.

    Parameters
    ----------
    backend : str
        "offline_replay" or "api".
    records : list[EncounterRecord] | None
        Required for offline_replay. Ordered by encounter window time.
    scenario_pack : str
        Filter applied to records (e.g., "maritime_chokepoints"). "all" disables.
    api_base_url : str | None
        Required for backend="api". Default http://127.0.0.1:8000/sim.
    step_penalty : float
        Per-step efficiency penalty.
    materialize_not_useful_penalty : float
        Penalty when accept action produces a not-useful materialization.
    refine_cost : float
        Cost of the secondary evidence pass.
    skip_missed_penalty : float
        Penalty when we skip a tile that would have been useful (only applies
        when offline replay has ground-truth `useful=True`).
    """

    # Gym-compatible metadata
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        backend: str = "offline_replay",
        records: Optional[list[EncounterRecord]] = None,
        scenario_pack: str = "all",
        api_base_url: Optional[str] = None,
        step_penalty: float = -0.01,
        materialize_not_useful_penalty: float = -0.10,
        refine_cost: float = -0.05,
        skip_missed_penalty: float = -0.01,
        seed: Optional[int] = None,
        tile_encoder=None,  # optional TileEncoder — when set, _tile_to_obs returns embedding
    ) -> None:
        self.backend = backend
        self.scenario_pack = scenario_pack
        self.tile_encoder = tile_encoder
        self.api_base_url = api_base_url or "http://127.0.0.1:8000/sim"
        self.step_penalty = step_penalty
        self.materialize_not_useful_penalty = materialize_not_useful_penalty
        self.refine_cost = refine_cost
        self.skip_missed_penalty = skip_missed_penalty
        self._rng = np.random.default_rng(seed)

        if backend == "offline_replay":
            if not records:
                raise ValueError("offline_replay backend requires non-empty records list")
            filtered = [r for r in records if scenario_pack in ("all", r.scenario_pack)]
            if not filtered:
                raise ValueError(
                    f"No records match scenario_pack={scenario_pack!r}. "
                    f"Available packs: {sorted({r.scenario_pack for r in records})}"
                )
            self._records = filtered
        elif backend == "api":
            self._records = []
            logger.warning(
                "SimSatEnv api backend is stubbed. Wire HTTP calls against "
                "%s/encounter/* before deployment.", self.api_base_url
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}; expected offline_replay or api")

        # Per-episode state (populated by reset())
        self._record_idx: int = 0
        self._current_record: Optional[EncounterRecord] = None
        self.current_window: int = 0
        self.window_history: list[dict] = []
        self._total_steps: int = 0

    # ---- Gym-style API ------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Return initial observation; reset episode state."""
        self._record_idx = 0
        self._total_steps = 0
        self.current_window = 1
        self.window_history = []

        if self.backend == "api":
            # TODO: POST /sim/encounter/reset and parse initial observation
            raise NotImplementedError("api backend reset not wired yet")

        self._current_record = self._records[self._record_idx]
        return self._tile_to_obs(self._current_record.tile)

    def step(self, action_int: int) -> tuple[np.ndarray, float, bool, dict]:
        """Apply a trust action; return (next_obs, reward, done, info)."""
        if self._current_record is None:
            raise RuntimeError("Must call reset() before step()")

        if action_int not in (TRUST_ACCEPT, TRUST_DEFER, TRUST_SKIP, TRUST_REFINE):
            raise ValueError(f"Invalid action {action_int}; expected 0..3")

        reward = self._compute_reward(self._current_record, action_int)
        self._total_steps += 1

        # Record boundary for TTT Scope 2
        self.window_history.append({
            "step": self._total_steps,
            "window": self.current_window,
            "window_id": self._current_record.window_id,
            "action": TRUST_ACTION_NAMES[action_int],
            "reward": reward,
            "target_id": self._current_record.target_id,
            "useful": self._current_record.useful,
        })

        self.current_window += 1
        self._record_idx += 1
        done = self._record_idx >= len(self._records)

        if done:
            next_obs = np.zeros((1, OBS_H, OBS_W), dtype=np.float32)
            self._current_record = None
        else:
            self._current_record = self._records[self._record_idx]
            next_obs = self._tile_to_obs(self._current_record.tile)

        info = {
            "window": self.current_window - 1,
            "window_id": self.window_history[-1]["window_id"],
            "action_name": TRUST_ACTION_NAMES[action_int],
            "episode_so_far": {
                "total_reward": float(sum(h["reward"] for h in self.window_history)),
                "windows_completed": self.current_window - 1,
            },
        }
        return next_obs, float(reward), bool(done), info

    def legal_actions(self) -> list[int]:
        """Trust action vocabulary. Constant for starter; extend for per-target selection."""
        return [TRUST_ACCEPT, TRUST_DEFER, TRUST_SKIP, TRUST_REFINE]

    def action_to_string(self, action_int: int) -> str:
        return TRUST_ACTION_NAMES.get(action_int, f"unknown({action_int})")

    def render(self) -> None:
        """Print compact ASCII summary of the current encounter window."""
        if self._current_record is None:
            print("[SimSatEnv] no active record (episode ended or not reset)")
            return
        r = self._current_record
        print(
            f"[SimSatEnv] window={self.current_window} target={r.target_id} "
            f"pack={r.scenario_pack} priority={r.target_priority} "
            f"cloud={r.sentinel_cloud_cover} elev={r.elevation_degrees}"
        )

    def close(self) -> None:
        self._current_record = None

    # ---- Internals ---------------------------------------------------------

    def _tile_to_obs(self, tile: np.ndarray) -> np.ndarray:
        """Normalize a raw Sentinel tile to (1, OBS_H, OBS_W) float32 in [0, 1]."""
        arr = np.asarray(tile, dtype=np.float32)

        # Handle shape variations: (H, W), (1, H, W), (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[0] > 1:
            # Multispectral → collapse to mean luminance for (1, H, W) starter shape
            arr = arr.mean(axis=0, keepdims=True)

        # Resize to (1, OBS_H, OBS_W) via nearest-neighbour.
        _, h, w = arr.shape
        if (h, w) != (OBS_H, OBS_W):
            ys = (np.arange(OBS_H) * h // OBS_H).astype(np.int32)
            xs = (np.arange(OBS_W) * w // OBS_W).astype(np.int32)
            arr = arr[:, ys[:, None], xs[None, :]]

        # Normalize to [0, 1]
        arr = arr / SENTINEL_NORM
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr.astype(np.float32)
        # LFM2.5-VL encoder path: returns (embed_dim,) embedding for MuZero FC network
        if self.tile_encoder is not None and getattr(self.tile_encoder, "embed_dim", None) is not None:
            return self.tile_encoder.encode(arr)
        return arr

    def _compute_reward(self, record: EncounterRecord, action_int: int) -> float:
        """Reward shaping — placeholder values; tune during stage-1 pretraining."""
        r = self.step_penalty

        if action_int == TRUST_ACCEPT:
            if record.useful is True and record.utility_realized is not None:
                r += float(record.utility_realized)
            elif record.useful is False:
                r += self.materialize_not_useful_penalty
            elif record.utility_realized is not None:
                # Useful unknown but utility recorded — credit it.
                r += float(record.utility_realized)
            else:
                r += 0.0  # unknown ground truth: neutral

        elif action_int == TRUST_DEFER:
            r += 0.0

        elif action_int == TRUST_SKIP:
            if record.useful is True:
                r += self.skip_missed_penalty

        elif action_int == TRUST_REFINE:
            r += self.refine_cost

        return float(r)


# --------------------------------------------------------------------------- #
# Convenience: build an env from the 86-trace corpus                          #
# --------------------------------------------------------------------------- #

def env_from_trace_corpus(
    trace_dicts: list[dict],
    scenario_pack: str = "all",
    **env_kwargs: Any,
) -> SimSatEnv:
    """
    Build a SimSatEnv from a list of stored-trace dicts (same shape the Phase 5
    T1 parity harness produces via `trace_to_inputs()`).

    Each dict must have at minimum: window_id, target_id, scenario_pack, tile.
    Optional: probe metadata, geometry, vla payload, outcome labels.
    """
    records = [_trace_dict_to_record(t) for t in trace_dicts]
    return SimSatEnv(backend="offline_replay", records=records,
                     scenario_pack=scenario_pack, **env_kwargs)


def _trace_dict_to_record(trace: dict) -> EncounterRecord:
    """Adapter: stored-trace dict → EncounterRecord."""
    sample = trace.get("sample", {}) or {}
    probe  = trace.get("probe", {}) or {}
    geom   = trace.get("geometry", {}) or {}
    obs    = trace.get("observation", {}) or trace.get("assessment", {}) or {}

    tile = trace.get("tile")
    if tile is None:
        # Fallback: zeros if no tile present (e.g., stub traces)
        tile = np.zeros((1, OBS_H, OBS_W), dtype=np.float32)
    else:
        tile = np.asarray(tile)

    return EncounterRecord(
        window_id=trace.get("trace_id") or trace.get("window_id", "unknown"),
        target_id=sample.get("target_id") or trace.get("target_id", "unknown"),
        scenario_pack=sample.get("scenario_pack", "all"),
        tile=tile,
        target_priority=sample.get("target_priority", "normal"),
        target_tags=list(sample.get("target_tags", [])),
        sentinel_available=probe.get("sentinel_available"),
        sentinel_cloud_cover=probe.get("sentinel_cloud_cover"),
        elevation_degrees=geom.get("elevation_degrees"),
        line_of_sight=geom.get("target_visible"),
        actual_action=obs.get("recommended_action") or trace.get("operator_action"),
        utility_realized=trace.get("utility_realized") or obs.get("usefulness_score"),
        useful=trace.get("useful") if trace.get("useful") is not None else obs.get("useful"),
    )
