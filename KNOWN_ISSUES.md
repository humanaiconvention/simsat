# SimSat Known Issues

Consolidated issue tracker. Originally authored 2026-04-21 from full-repo review; updated 2026-04-24 to reflect Phase 5 completion and multi-model backend additions; further updated 2026-04-24 (session 2) — items 4, 12, 13, 14 resolved; updated 2026-04-24 (session 3) — item 25 updated with kernel v5 results (training ran, loss flat, eval 2/10 shortlist); updated 2026-04-24 (session 4) — item 23 resolved (TTT viability wiring), notebook v3 written (Fix #13 permanent + Fix #14); updated 2026-04-25 (session 5) — items 7, 16 resolved (clip_local all cases, 4 pinned cases), notebook v3 pushed to Kaggle (running), 21-test smoke suite added, per-track pitch thesis written; updated 2026-04-25 (session 6) — item 24 resolved with substitution (HFVisionTowerEncoder + SigLIP-base default), collaborator-readiness pass landed + CI green on `codex/runtime-test-baseline` (commits 8fe720f → 8a66633), tech-debt cleanup (lifespan migration, encounter logger, ValueError 503→400); updated 2026-04-27 — items 6, 16, 18, 25 corrected after v5 DiLoCo diagnostic + per-layer LoRA audit revealed v1-v10 all trained zero language-model parameters (target_modules `.linear` suffix matched only multimodal towers); v11 with corrected regex is the first real fine-tune (item 25 superseded), reviewed pool grew 8 → 37 via batch_review.py (item 16 superseded), dataset bumped to 713 rows (item 18 superseded). Item 27 added for the partial-save quirk in v11 (k/v missing on layers 15-34).

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
| 6 | ObservationVLA usefulness-score MAE 0.27 — model under-confident vs. operator | `OBSERVATION_VLA_EVAL.md` | **Resolved with v11 (2026-04-27)** — first run with corrected `target_modules`. v9/v10 claimed MAE=0.16 on N=7, but those measurements were against a no-op LoRA adapter (= stock Gemma-4-E2B, see `notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`). v11 actual SimSat fine-tune at N=37: MAE=**0.13**, exact action agreement **0.86**, useful agreement **0.97**. The accept-bias is gone (32 of 37 cases match operator label exactly). |
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
| 16 | Only 3 operator-reviewed pinned cases; N=3 is noisy for model claims | `SUBMISSION_CASEBOOK.md` | **Resolved** — N=3 → N=4 (schema migration, SF Bay pinned) → **N=37** (2026-04-27 review session via `scripts/batch_review.py`, see `notebooks/DILOCO_ROUND0_RUN_NOTE.md`). Operator-review pool now covers all 10 challenge targets with at least 1 reviewed case each (most have 3-5). Casebook still shows the 4 originally-pinned cases; the broader 37-case pool drives the eval (`OBSERVATION_VLA_EVAL.md`). |

---

## Tier 4 — Entry-Specific

| # | Issue | Location | Status |
|---|---|---|---|
| 17 | Per-track one-sentence thesis for the pitch video | `CHALLENGE_ENTRY.md` Two-Track section | **Open** — track framing is locked; one-liner video copy drafted at recording time |
| 18 | Entry B fine-tune dataset design | `datasets/simsat-gemma4-v1/` | **Resolved** — bumped from 294 → **713 weighted ChatML rows** (2026-04-27) after the batch_review session and `REFINE_BOOST=1.5`. `multimodal_reviewed` (weight 6-8) jumped 4 → 34. Dataset slug unchanged: `benhaslam/simsat-gemma4-v1` (now v2 on Kaggle). |
| 19 | Entry B VLA backend adapter | `src/sim/observation_vla/` | **Resolved** — `TransformersVLMAdapter` (generic), `GenesisAdapter`, `TesseractT3Adapter`, `backend_factory.py` all shipped |
| 20 | Google Drive OAuth scope too narrow — blocks LFM2.5 Colab notebook recon | platform integration | **Open / Low priority** — not on critical path |
| 21 | VLA-layer TTT wiring | `adapter.py` | **Resolved architecturally** — `vla_online_update()` implemented; tunes confidence blend weights from operator usefulness_score |
| 22 | Trust-layer TTT wiring | `src/sim/encounter/trust_model.py` | **Resolved architecturally** — `online_update()` implemented and wired via `_apply_trust_layer_ttt()` in service.py |
| 23 | Viability gate integration with TTT adaptation streams | `src/sim/haic/viability.py` | **Resolved** — `evaluate_ttt_viability(trust_snapshot)` added to viability.py (3 TTT gates: weight_drift, update_rate, error_bias). Wired into `_apply_trust_layer_ttt()` in service.py after `online_update()`; failed gates emit WARNING log, do not block outcome registration. |
| 24 | LFM2.5-VL encoder — actual weights | `src/sim/muzero/tile_encoder.py` | **Resolved (2026-04-27)** — Liquid AI shipped public LFM2.5-VL weights on 2026-04-11 ([LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)). `HFVisionTowerEncoder` updated to detect `model_type="lfm2_vl"`, load via `AutoModelForImageTextToText`, and route the encode pass through `model.model.vision_tower` (Siglip2VisionModel, 768-dim under the LFM wrapper). `build_encoder("lfm2vl")` now defaults to `LiquidAI/LFM2.5-VL-450M` (450M params, sub-250ms edge inference per Liquid AI; vision tower is SigLIP-2 NaFlex shape-optimized 86M). Local code-path sanity check via `init_empty_weights` verified `vision_tower` resolves correctly and `embed_dim=768`. Real-weight load + encode round-trip + `MUZERO_LFM_EVAL.md` re-run pending (~900 MB download for 450M, longer for 1.6B). SigLIP-base remains the safe offline default for `HFVisionTowerEncoder()` with no model_id; `build_encoder("lfm2vl")` is the explicit LFM path. |
| 25 | Kaggle Gemma-4-E2B v2 kernel — gradient flow | `notebooks/kaggle-simsat-gemma4-v1/` | **CORRECTED 2026-04-27.** Earlier "v9 / v10 resolved" claims were wrong. v5 DiLoCo diagnostic + per-layer adapter audit showed v1-v10 all trained zero language-model parameters: `target_modules=["q_proj.linear", ...]` matched only Gemma-4's vision/audio towers (which wrap projections in `Gemma4ClippableLinear` with a `.linear` sub-module); the language model decoder layers expose `q_proj` directly. Every v1-v10 adapter has 224 vision + 72 audio LoRA tensors, 0 language, all `lora_B = 0.0` — gradient flow worked, but PEFT had no language-model LoRA modules to update. Eval numbers measured stock `google/gemma-4-E2B-it`. **v11 (2026-04-27, kernel version 11)** ships the corrected anchored regex `r"model\.language_model\.layers\.\d+\.(self_attn\|mlp)\.(q\|k\|v\|o\|gate\|up\|down)_proj$"`, plus a post-save sanity gate that fails the kernel loudly on null-LoRA. v11 audit: 410 LoRA tensors, 100% `language_model`, lora_B 205/205 non-zero, training_loss 0.2429. Eval at N=37: MAE 0.13, exact 0.86, bucketed 0.86, useful 0.97. See `notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`. |
| 26 | Genesis + Tesseract T3 model weights | `src/sim/observation_vla/genesis_local.py`, `tesseract_t3_local.py` | **Genesis adapter live (2026-04-26, session 8)** — `genesis_local.py` rewritten as native loader using orchOSModel-genesis-v3 directly (no HF dependency). Loads `genesis_152m_instruct.safetensors` via safetensors metadata config + NeoX tokenizer + ChatML format. Model loads cleanly on BEAST (151.76M params, GLA+FoX hybrid). Zero-shot JSON output unreliable at 152M — Guilherme needs to fine-tune on simsat-gemma4-v1 dataset (same as Gemma-4 track) or provide a working system prompt. Setup: `GENESIS_REPO_PATH=/path/to/orchOSModel-genesis-v3 GENESIS_WEIGHTS_PATH=/path/to/genesis_152m_instruct.safetensors OBSERVATION_VLA_BACKEND=genesis`. Tesseract T3 still pending Garrett. **Improved error surfacing (2026-04-27)** — Genesis silent stub_fallback now logs the actual load failure; Tesseract T3 error tags rewrite `OBSERVATION_VLM_*` → `TESSERACT_T3_*` so users see the env var they actually configure. |
| 27 | v11 adapter partial save — k_proj/v_proj LoRA missing for layers 15-34 | `notebooks/kaggle-simsat-gemma4-v1/simsat_gemma4_v1_training.py` save path | **Open** — v11 saves only 410 of 490 expected LoRA tensors. Layers 0-14 have all 7 modules; layers 15-34 are missing k_proj and v_proj. Local `get_peft_model` against `init_empty_weights` matches all 245 modules — bug is at Kaggle save time, not config time. Most-likely cause: Gemma-4-E2B GQA (`num_key_value_heads=1`) + tied k/v across layer groups + PEFT `state_dict` dedupe by tensor identity. PEFT zero-fills missing keys at load, so v11 is functional but partially trained. Audit + diagnosis in `notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md` (Follow-up section). Cheap diagnostic: v12 with `peft>=0.20` and a tightened sanity gate that requires all 245 LoRA pairs. |

---

## Conventions

- **Resolved** = addressed in codebase, no further action required
- **Resolved architecturally** = implemented at the design level; full live demonstration may require prize hardware (NVIDIA Orin 16GB)
- **Open** = not yet addressed; evaluate against deadline priority
- **Pending** = action required by a specific person or event
- **Accepted** = known limitation, impact assessed as acceptable
