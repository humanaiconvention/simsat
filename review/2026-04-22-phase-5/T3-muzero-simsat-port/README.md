# T3 — MuZero-SimSat port starter

Ports the ARC3 MuZero pipeline (`humanaiconvention/arc3`) to the SimSat encounter-planning loop, so stacked test-time training (Spec #21) can run on the same proven machinery we already use on Kaggle for ARC-AGI-3.

## Why MuZero for SimSat

Three findings from reading the ARC3 pipeline (pretrain + finetune + TTT notebooks + `arc3_game.py`):

1. **MuZero is actually a better on-orbit fit than LFM2.5 / Gemma-4 for the Liquid & General AI tracks' hardware envelope.**
   - MuZero ResNet: 3 blocks × 64 channels + heads = a few MB of weights.
   - LFM2.5-VL / Gemma-4: 2–8 GB of weights.
   - Prize budget: 5 MB uplink / 10 MB downlink. MuZero fits; LLM weights don't.
   - MCTS + 3-block ResNet on NVIDIA Orin 16GB is genuinely fast; LLM inference at every encounter window blows the GPU-hour budget.

2. **The ARC3 three-stage chain maps directly onto SimSat's training needs.**
   | Stage | ARC3 | SimSat |
   |---|---|---|
   | 1 | MuZero on Atari-HEAD + ARC3 recordings | MuZero on tile recordings (pre-SimSat work) + 86-trace Sentinel corpus |
   | 2 | Warm-start, fine-tune on ARC3 alone with 40× augmentation | Warm-start, fine-tune on SimSat scenario-pack traces with matching augmentation |
   | 3 | Two-scope TTT during live Arcade play | Two-scope TTT during live SimSat Django-backend play |

3. **MuZero's encounter-window planning matches our pitch framing.** The pitch already calls SimSat "encounter-planning not continuous propagation"; MuZero's `h / g / f` + MCTS over imagined future states is literally an encounter planner. The framing tightens, not stretches.

## What this bundle ships

| File | Role | Lines | LOC anchor |
|---|---|---|---|
| `simsat_env.py` | Gym-style wrapper around the encounter loop. Offline-replay backend (for training) + API backend stub (for deployment). | ~290 | — |
| `simsat_game.py` | MuZero `AbstractGame` adapter + `MuZeroConfig` + `TTTScope1` + `TTTScope2`. Direct port of `arc3_game.py` with SimSat-appropriate defaults. | ~290 | ported from `arc3_game.py` |
| `scripts/smoke_test_simsat_env.py` | Zero-dependency pytest-style harness verifying env, rewards, action vocabulary, scenario-pack filter, MuZeroConfig, and TTT budget decay. | ~220 | — |
| `README.md` | This file. | — | — |

All three Python files have been **syntax-checked and runtime-smoke-tested locally** before commit:

```
[1] Env reset and step
  ✓ reset → obs (1, 64, 64) float32 range [0.000, 0.250]
  ✓ stepped through 6 records, total_reward=+1.397, done at step 5
[2] Reward shaping
  ✓ accept+useful: reward=+0.940 (~0.94 expected)
  ✓ accept+not_useful: reward=-0.110 (~-0.11 expected)
  ✓ refine: reward=-0.060 (~-0.06 expected)
[3] Legal actions and invalid rejection
  ✓ legal_actions=[0, 1, 2, 3], invalid action correctly rejected
[4] Scenario-pack filter
  ✓ scenario_pack filter: 2 records matched 'maritime_chokepoints'
[5] SimSatGame + TTT scopes
  ✓ MuZeroConfig: obs=(1, 64, 64) action_space=[0, 1, 2, 3] network=impala
  ✓ SimSatGame: reset → obs (1, 64, 64), legal_actions=[0, 1, 2, 3]
  ✓ TTTScope1 budgets: [32, 20, 13, 8] (base 32, decay 0.65)
  ✓ TTTScope2 budgets: [16, 11, 7, 5] (base 16, decay 0.70)
  ✓ window_history has 4 entries as expected
```

## What's directly ported vs. adapted

**Ported verbatim from `arc3_game.py`:**
- `MuZeroConfig` structure — all hyperparameters, IMPALA network shape, support size, MCTS depth
- `TTTScope` / `TTTScope1` / `TTTScope2` — identical budget-decay schedules (BASE=32, DECAY=0.65 for Scope 1; BASE=16, DECAY=0.70 for Scope 2)
- `AbstractGame` adapter shape (`reset` / `step` / `legal_actions` / `render` / `close` / `action_to_string`)
- Lazy import pattern for optional dependencies (arc3 uses arcengine; SimSat uses torch)

**Adapted for SimSat:**
- Action space: `range(18)` (ARC3 click grid + multi-action) → `range(4)` (accept / defer / skip / refine)
- Level boundary concept: ARC3 level completion → SimSat encounter window. `current_level` → `current_window`, `level_history` → `window_history`.
- Observation source: `arcade.Arcade().make(game_id).reset().frame` → Sentinel tile from `EncounterRecord.tile` normalized to [0, 1].
- Normalization constant: `_MAX_COLOR = 12.0` → `SENTINEL_NORM = 10_000.0` (Sentinel-2 L2A reflectance scaling).
- Reward shaping: ARC3's `+1/level, +10/win, -0.01/step` → SimSat's `+utility_realized/useful-accept, -0.10/not-useful-accept, -0.05/refine, -0.01/step, -0.01/skip-when-useful`.

**New for SimSat:**
- `EncounterRecord` dataclass — the offline-replay tuple (tile + target metadata + probe + geometry + optional outcome labels). Shape matches what the Phase 5 T1 parity harness already extracts from stored traces via `trace_to_inputs()`.
- `env_from_trace_corpus()` — convenience constructor that bridges the existing 86-trace store format into `EncounterRecord` list.
- `backend="api"` — stub for the live SimSat Django backend (analog of ARC3's `arcade.Arcade`); HTTP wiring TODO before stage-3 deployment.

## How this fits with Phase 5 T1 (WCLITrustModel)

Two architectural options for integrating MuZero with the existing trust layer:

**Option A (recommended for starter): MuZero outputs the action; WCLI audits.**
- MuZero's policy head directly outputs one of {accept, defer, skip, refine}.
- `WCLITrustModel.decide()` is consulted as a symbolic overlay: it re-expresses the MuZero action in the canonical reason-code vocabulary, and can hard-override in corner cases (e.g., line_of_sight=False forces skip regardless of MuZero's preference).
- Trust-layer TTT (Spec #22) still fires from realized-utility feedback, but its role shifts from "pick the action" to "calibrate the symbolic gate around MuZero's policy."

**Option B (richer, later): MuZero outputs lower-level actions; WCLI interprets.**
- MuZero's policy head operates over a larger action space (e.g., target selection × trust verdict).
- WCLI aggregates MuZero's distribution + contextual signals into the final trust decision.
- More expressive but adds modeling complexity. Leave for post-submission.

T3 ships with Option A in mind — the `SimSatEnv` action space is the 4 trust actions directly.

## Application workflow

1. **Copy files into `D:\SimSat`:**
   ```bash
   cp T3-muzero-simsat-port/simsat_env.py  D:\SimSat\src\sim\muzero\simsat_env.py
   cp T3-muzero-simsat-port/simsat_game.py D:\SimSat\src\sim\muzero\simsat_game.py
   cp T3-muzero-simsat-port/scripts/smoke_test_simsat_env.py D:\SimSat\scripts\smoke_test_simsat_env.py
   ```
   (Create `src/sim/muzero/` directory if missing.)

2. **Run the smoke test:**
   ```bash
   python scripts/smoke_test_simsat_env.py
   ```
   Expect all 5 tests green, same as the pre-commit run.

3. **Bridge the existing 86-trace corpus.** Write `build_encounter_records.py`:
   - Load the 86 stored ObservationVLA traces (same source as Phase 5 T1's parity harness).
   - Convert each to `EncounterRecord` using `_trace_dict_to_record()` (or a local variant if the trace shape differs).
   - Save as a pickled list or pass directly to `SimSatEnv(records=...)`.

4. **Clone `werner-duvaud/muzero-general` and the `humanaiconvention/arc3` patches:**
   ```bash
   git clone https://github.com/werner-duvaud/muzero-general.git D:\SimSat\external\muzero-general
   git clone https://github.com/humanaiconvention/arc3.git        D:\SimSat\external\arc3
   cp D:\SimSat\external\arc3\muzero_patch\models.py D:\SimSat\external\muzero-general\models.py
   ```
   `simsat_game.py` already expects `external/muzero-general/games/abstract_game.py` relative to its own parent dir.

5. **Run stage-1 pretraining on Kaggle.** Clone the ARC3 pretrain notebook (`arc3_muzero_pretrain.ipynb`), swap the data-mix cell to use the SimSat encounter corpus instead of Atari+ARC3, and run. Produces `simsat_stage1.checkpoint`.

6. **Run stage-2 finetune.** Clone the ARC3 finetune notebook. Point at `simsat_stage1.checkpoint` as the stage-1 ref. Augmentation pipeline (crop, hflip, vflip, window, NN-only) transfers directly to Sentinel tiles.

7. **Run stage-3 TTT.** Clone the ARC3 TTT notebook. Point at `simsat_stage2.checkpoint`. Swap `Arcade` for `SimSatEnv(backend="api", api_base_url="http://localhost:8000/sim")` once the API backend is wired.

## Open decisions and blockers

1. **Liquid Track eligibility with MuZero-only architecture.**
   The track rewards "LFM2 fine-tuning + public weights + training code." Three reconciliations, pick one:
   - (a) Distill LFM2.5 → MuZero (LFM2.5 is fine-tuned as teacher; MuZero is student).
   - (b) Hybrid: LFM2.5-VL becomes the observation encoder, feeding MuZero's `h()`. LFM2.5 is LoRA-fine-tuned.
   - (c) Cite LFM2 for methodology; accept the Liquid Track eligibility risk.
   Logged in KNOWN_ISSUES. Ben × Guilherme owns this decision.

2. **`benhaslam/kaggle-simsat-gemma4-v3` relationship to stage-2.**
   That Kaggle kernel (2026-04-13) appears to be an earlier SimSat-specific Gemma-4 training run. It could serve as the Gemma-4 lineage for the General AI Track's stage-2 (teacher distillation or trace-augmentation source). Clarifying its status unblocks the exact lineage claim in the pitch.

3. **API backend not yet wired.**
   `SimSatEnv(backend="api")` raises `NotImplementedError` on `reset()`. Wire HTTP calls against `/sim/encounter/*` endpoints before stage-3 live deployment. Synchronous `requests` or async `aiohttp` both work; pick whatever the existing SimSat Python clients use.

4. **TTT budget decay tuning.**
   Default `DECAY_FACTOR = 0.65` floors to 1 step after ~8 windows. ARC3 games have ~7 levels so this fits. SimSat scenario evals can have 30+ encounter windows per pass — late windows would TTT with 1 step. Either raise `DECAY_FACTOR` to 0.9 or implement the surprise-driven adjustment referenced in the `TTTScope.ttt_budget` docstring (loss-driven, not fixed schedule).

5. **Observation channel count.**
   Starter uses (1, 64, 64) single-channel to clean-port from ARC3. Multi-spectral (C=3 RGB or C=13 full S2) is a clear rubric upgrade — see "Use of Satellite Imagery: multi-spectral rewarded" in the event rubric. Widen `OBS_C`, bump IMPALA input channels, re-pretrain. This is a stretch item.

## Next phases this unblocks

- **Phase 5 T4 (proposed):** bridge the 86-trace corpus into `EncounterRecord` list; wire the Django API backend; run stage-1 pretrain on Kaggle.
- **Phase 6 (proposed):** stage-2 fine-tune + TTT deployment run, with concrete scorecard comparison (frozen stage-2 vs stage-3 TTT on held-out scenario packs). This is the "measurable improvement over base" deliverable for the Liquid Track.
- **Phase 7 (proposed):** six-gate viability module (Spec #23) wiring into both TTT adaptation streams. Covers the "safety" claim in the pitch.

## Why this is a "starter" not a "finished port"

- The MuZero trainer + replay-buffer + Ray orchestration is NOT duplicated here — it lives in `werner-duvaud/muzero-general` (installed via git clone in the Kaggle notebooks) and only needs `SimSatGame` to function. Don't re-implement; just clone the upstream.
- Checkpoint format is identical to ARC3's (pickled dict with `weights / config / stage / training_steps`), so transfer from ARC3 notebooks is one line.
- The API backend is deliberately a stub — wiring it requires real HTTP calls against a running Django instance, which is local-only.
- Multi-spectral is deferred.
- Tuning of TTT budgets, reward shaping constants, and action-space extension is all deferred to the first real training run where we have signal to tune against.

Ship T3, then iterate.
