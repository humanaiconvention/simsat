# SimSat Known Issues

Consolidated issue tracker. Originally authored 2026-04-21 from full-repo review; updated 2026-04-24 to reflect Phase 5 completion and multi-model backend additions; further updated 2026-04-24 (session 2) — items 4, 12, 13, 14 resolved; updated 2026-04-24 (session 3) — item 25 updated with kernel v5 results (training ran, loss flat, eval 2/10 shortlist); updated 2026-04-24 (session 4) — item 23 resolved (TTT viability wiring), notebook v3 written (Fix #13 permanent + Fix #14); updated 2026-04-25 (session 5) — items 7, 16 resolved (clip_local all cases, 4 pinned cases), notebook v3 pushed to Kaggle (running), 21-test smoke suite added, per-track pitch thesis written; updated 2026-04-25 (session 6) — item 24 resolved with substitution (HFVisionTowerEncoder + SigLIP-base default), collaborator-readiness pass landed + CI green on `codex/runtime-test-baseline` (commits 8fe720f → 8a66633), tech-debt cleanup (lifespan migration, encounter logger, ValueError 503→400).

Deadline: **Friday May 8, 2026, 5:00 PM PDT**

---

## Tier 1 — Correctness

| # | Issue | Location | Status |
|---|---|---|---|
| 1 | Naive `datetime.fromtimestamp()` emits local-time ISO strings → phase-shifted encounter windows in any non-UTC environment | `src/sim/simulator.py:111` | **Resolved** — `tz=datetime.timezone.utc` already present at line 111 |
| 2 | Stray `tmp*_simsat_api.py` files at `src/sim/` root | `src/sim/` | **Resolved** — no such files found as of 2026-04-24 |
| 3 | Duplicate standalone `gui.py` with hardcoded LAN IP | repo root | **Resolved** — root `gui.py` is the canonical copy; no duplicate found |
| 4 | Pinned casebook reviewer notes undermine the reviewer | `SUBMISSION_CASEBOOK.md` | **Resolved** — parenthetical asides removed from Houston and Suez review notes |

---

## Tier 2 — Thesis + AI Story

| # | Issue | Location | Status |
|---|---|---|---|
| 5 | All 3 pinned cases were `accept→accept`; thesis `accept→refine` was rhetorical | `SUBMISSION_CASEBOOK.md` | **Resolved** — Port of Rotterdam case (`urban_coastal_ambiguity`, trace `trace_4f65355f`) is scaffold `accept`, trust `refine`, operator confirmed `refine` (cloud 48.78%) |
| 6 | ObservationVLA usefulness-score MAE 0.27 — model under-confident vs. operator | `OBSERVATION_VLA_EVAL.md` | **Open (documented)** — disclosed as calibration gap in CHALLENGE_ENTRY.md and SUBMISSION_BRIEF.md; downstream actions consume binary agreement not raw score |
| 7 | Pinned traces showed `Observation runtime: stub`; backend is `clip_local` | `SUBMISSION_CASEBOOK.md` | **Resolved** — all 4 pinned cases now show `clip_local`. torchvision reinstalled (0.22.0+cu118 matched to torch 2.7.0+cu118), CLIP loads cleanly. `scripts/reassess_pinned_traces.py` re-ran Houston + Suez + SF Bay; Rotterdam was already clip_local. |
| 8 | HAIC narrative stance | pitch docs + dashboard | **Resolved** — mechanism-only stance locked in; HAIC brand absent from submission pitch; six viability gates described mechanically |
| 9 | Stacked TTT thesis | pitch docs | **Resolved** — architecturally described in CHALLENGE_ENTRY.md + SUBMISSION_BRIEF.md; both TTT streams (VLA + trust-layer) documented with viability gate gating |

---

## Tier 3 — Rigor + Polish

| # | Issue | Location | Status |
|---|---|---|---|
| 10 | Sweep generation: `quality_skips=60, added_distinct=0` | `REVIEW_SET_BUILD.md` | **Resolved** — root cause was `_is_model_backed()` requiring `runtime_mode="clip_local"` exactly; fixed to accept any non-stub mode |
| 11 | 460/546 traces are safe-prune `stub_fallback` cruft | ObservationVLA corpus | **Open** — prune script documented; run `scripts/observation_vla_prune_fallback.py` when corpus size matters |
| 12 | `DJANGO_DEBUG=1`, `ALLOWED_HOSTS=["*"]`, `CORS_ALLOW_ALL_ORIGINS=True` in dashboard | `src/dashboard/sat_dashboard/settings.py` | **Resolved** — all three default to secure/off; env-var opt-in for local dev |
| 13 | No propagator unit test against known TLE-epoch expected output | `src/sim/` | **Resolved** — `src/sim/tests/test_propagator.py` added; tests altitude plausibility, determinism, time advancement, and coerce type coverage |
| 14 | `flatted.py` may be dead code in dashboard | `src/dashboard/` | **Resolved** — file lives only in `node_modules/flatted/python/`; no project code imports it; already gitignored |
| 15 | `WindowDetector` uses spherical Earth (R=6371 km) vs. WGS84 propagator | `src/sim/encounter/windows.py` | **Accepted** — ≤0.2° elevation error at 30° threshold; no impact on challenge scoring |
| 16 | Only 3 operator-reviewed pinned cases; N=3 is noisy for model claims | `SUBMISSION_CASEBOOK.md` | **Resolved** — schema migrated: `observation_submission_cases` PK changed from `scenario_pack` → `trace_id` (migration runs automatically at startup). SF Bay pinned as 4th case. Casebook now shows **4 human-reviewed cases** (Houston, Suez, Rotterdam, San Francisco Bay), all `clip_local`. |

---

## Tier 4 — Entry-Specific

| # | Issue | Location | Status |
|---|---|---|---|
| 17 | Per-track one-sentence thesis for the pitch video | `CHALLENGE_ENTRY.md` Two-Track section | **Open** — track framing is locked; one-liner video copy drafted at recording time |
| 18 | Entry B fine-tune dataset design | `datasets/simsat-gemma4-v1/` | **Resolved** — 294 weighted ChatML rows; `benhaslam/simsat-gemma4-v1` on Kaggle |
| 19 | Entry B VLA backend adapter | `src/sim/observation_vla/` | **Resolved** — `TransformersVLMAdapter` (generic), `GenesisAdapter`, `TesseractT3Adapter`, `backend_factory.py` all shipped |
| 20 | Google Drive OAuth scope too narrow — blocks LFM2.5 Colab notebook recon | platform integration | **Open / Low priority** — not on critical path |
| 21 | VLA-layer TTT wiring | `adapter.py` | **Resolved architecturally** — `vla_online_update()` implemented; tunes confidence blend weights from operator usefulness_score |
| 22 | Trust-layer TTT wiring | `src/sim/encounter/trust_model.py` | **Resolved architecturally** — `online_update()` implemented and wired via `_apply_trust_layer_ttt()` in service.py |
| 23 | Viability gate integration with TTT adaptation streams | `src/sim/haic/viability.py` | **Resolved** — `evaluate_ttt_viability(trust_snapshot)` added to viability.py (3 TTT gates: weight_drift, update_rate, error_bias). Wired into `_apply_trust_layer_ttt()` in service.py after `online_update()`; failed gates emit WARNING log, do not block outcome registration. |
| 24 | LFM2.5-VL encoder — actual weights | `src/sim/muzero/tile_encoder.py` | **Resolved with substitution (2026-04-25, session 6)** — `HFVisionTowerEncoder` class added; default `model_id="google/siglip-base-patch16-224"` (768-dim). SigLIP load + encode round-trip verified on BEAST (`RUN_HF_TILE_ENCODER_TESTS=1 pytest src/sim/tests/test_tile_encoder.py::TestHFVisionTowerEncoderReal`). When/if Liquid AI ships public LFM2.5-VL weights, change `model_id` and `embed_dim` in `build_encoder("lfm2vl", ...)` callers — the integration contract is unchanged. `LFM2VLEncoderStub` remains as a back-compat alias subclass of HFVisionTowerEncoder. 6 new offline factory tests + 2 opt-in real-model tests. |
| 25 | Kaggle Gemma-4-E2B v2 kernel — gradient flow | `notebooks/kaggle-simsat-gemma4-v1/` | **In progress (v4 kernel running 2026-04-25)** — v5 run: loss flat 3.9454 → root cause = prompt-token gradient dilution (90% non-signal tokens). v3 notebook applied Fix #14 (DataCollatorForCompletionOnlyLM) + Fix #13 (bfloat16) + torchao uninstall + peft_config guard. v3 kernel errored: `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'` — TRL 0.15.0 removed the class. **Fix #15 (v4 notebook)**: pin `trl>=0.12.0,<0.15.0` + defensive try/except import fallback. v4 pushed 2026-04-25; kernel `benhaslam/simsat-gemma4-v1-training` v4 RUNNING. Expect loss to descend this run. |
| 26 | Genesis + Tesseract T3 model weights | `src/sim/observation_vla/genesis_local.py`, `tesseract_t3_local.py` | **Pending collaborators** — adapters wired, weights pending Guilherme + Garrett; see `COLLABORATOR_GUIDE.md` |

---

## Conventions

- **Resolved** = addressed in codebase, no further action required
- **Resolved architecturally** = implemented at the design level; full live demonstration may require prize hardware (NVIDIA Orin 16GB)
- **Open** = not yet addressed; evaluate against deadline priority
- **Pending** = action required by a specific person or event
- **Accepted** = known limitation, impact assessed as acceptable
