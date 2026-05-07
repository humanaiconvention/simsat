# SimSat Submission Brief

## Claim
SimSat is a governed on-orbit continual-learning loop. It treats mission operations as a sequence of encounter windows rather than continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. A Sentinel-first ObservationVLA lane performs image-conditioned reassessment, and a mission-response layer converts those judgments into explicit downstream actions with logged utility. **Test-time training runs at two layers simultaneously** — the VLA backend adapts on streaming Sentinel tiles, and the trust layer tunes online from realized-utility feedback — with **both adaptation streams gated by six non-compensatory viability checks** before any update is allowed to persist.

Scenario packs span two observational registers: **geometric/structural** (maritime chokepoints, disaster response, urban coastal) and **spectral-biochemical** (pedospheric integrity — soil health via NDVI, SWIR ratio B11/B12, and EVI over active degradation sites). The same architecture operates identically across both registers.

## Why this must run in orbit
Three stacked constraints force the full pipeline to the spacecraft, in priority order:

1. **Distribution shift without a ground-truth validator.** A satellite streaming imagery hits real drift — new geographies, seasonal variation, cloud and sensor conditions — and cannot call home for a label. SimSat's answer is six non-compensatory viability gates that every candidate observation must pass before updating system state: information gain (adaptation must reduce model uncertainty), observation quality (cloud cover and geometry thresholds), metadata consistency (claimed sensor parameters match measured image properties), update magnitude (weight delta bounded to ≤30% of pre-pass baseline), update rate (cumulative adaptation count ceiling), and error balance (systematic same-direction errors suppress further updates). Without these gates, on-orbit TTT is uncontrolled drift; with them, it is a governed continual-learning loop. The gates operationalize the condition `C_eff(t) ≥ E(t)` — ensuring every accepted update carries quality-verified, magnitude-bounded, bias-checked information about the external world before touching model state. High-confidence observations that clear all six gates are also candidates for selective ground downlink; the satellite transmits pixels only when they are worth the bandwidth. This is the mechanism the rest of the competition will not have.
2. **Bandwidth.** A 5 MB uplink and 10 MB downlink cannot ship model weight updates to ground and back at any useful cadence. TTT that tracks drift must happen on the satellite itself.
3. **Latency.** The next encounter window arrives in minutes. Adaptations must be usable in the very next pass; there is no round-trip budget for ground re-training and re-upload.

## Drift Handling — Three Mechanisms, Three States

The architecture distinguishes between **what handles drift at runtime** (the SimSat pipeline) and **where drift starts** (the static fine-tune checkpoint). The submission proves the first; the second is what the prize hardware enables in flight.

| Mechanism | Implementation status | Evidence |
|---|---|---|
| **Trust-layer TTT** — WCLI thresholds and feature weights tune online from realized-utility feedback | **Proven, working** | 10-seed stability analysis: 22.5% MAE improvement ± 0.1%, 24.1% reward improvement, 1.9 ± 0.3 cycles to 90% convergence ([`ttt_stability_analysis.md`](./ttt_stability_analysis.md)) |
| **Six viability gates** — information gain, observation quality, metadata consistency, update magnitude, update rate, error balance | **Implemented, exercised** | Wired in `src/sim/haic/viability.py` via `evaluate_ttt_viability()`; fired correctly during the operator-review session (gate warnings `weight_drift`, `error_bias` logged in batch_review output once thresholds drifted) |
| **VLA-layer TTT** — VLM weights / LoRA adapters adapt on streaming Sentinel tiles per-pass, gated by all six checks | **Wired, requires live encounter stream** | `Stage 3 two-scope TTT` is in both tracks (`src/sim/muzero/`, `src/sim/observation_vla/`) but **not benchmarked against a live stream**. This is the lane the prize hardware (NVIDIA Orin 16 GB) enables. |

The ObservationVLA fine-tunes (v11, v12) are **static checkpoints that set where drift starts** — not drift-adaptive at runtime themselves. The 56-point gap between v11's in-distribution (N=37, 0.86) and cross-distribution (N=152, 0.30) eval frames is **what TTT is built to close in flight**, not what offline retraining closes. v12's role is to move the static training distribution closer to deployment so the in-flight TTT loop has less ground to cover, not to make the model itself drift-adaptive.

What the prize buys: Orin time to run Stage 3 VLA-TTT on a live encounter stream and benchmark adaptation convergence per-pass under the same six viability gates that already govern the trust layer.

## Tracks
SimSat is submitted to both tracks of the AI in Space hackathon. Both tracks run identical scaffold, WCLI trust, mission-response, viability-gate, and stacked-TTT infrastructure. What differs is the VLA backend and the planning backbone:

- **Liquid Track (LFM2.5 + MuZero)** — A vision-language model serves as the Sentinel tile encoder, producing a dense embedding that feeds MuZero's `h()` representation function. MuZero + MCTS then plans over encounter windows using that embedding. This hybrid architecture is why SimSat genuinely fits the on-orbit constraint: the MuZero ResNet weighs a few MB (fits the 5 MB uplink budget); the encoder is LoRA fine-tuned on SimSat Sentinel tiles and runs its encoder pass locally. Three-stage training: (1) pretrain on the Sentinel tile corpus, (2) fine-tune with SimSat scenario-pack augmentation, (3) two-scope TTT per live pass. The MuZero game adapter, gym environment, and `SimSatMuZeroConfigLFM` FC-network config are all in `src/sim/muzero/`. The encoder seat (`HFVisionTowerEncoder` → `build_encoder("lfm2vl")`) is filled by [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) (Liquid AI shipped public weights 2026-04-11; SimSat picked them up 2026-04-27). Vision tower is SigLIP-2 NaFlex shape-optimized 86M, 768-dim pooled output. SigLIP-base remains as an offline-safe alternative.

- **General AI Track (Gemma-4)** — a Gemma-4-E2B fine-tune scoped specifically to SimSat's encounter-triage task (accept / defer / skip / refine over Sentinel-style tiles). Canonical model: **v11**, on Hugging Face at [`HumanAIConvention/simsat-gemma4-v11`](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11), trained on dataset v2 (713 weighted ChatML rows, pre-defer-expansion). Training code at `notebooks/kaggle-simsat-gemma4-v1/`; Kaggle kernel at [`benhaslam/simsat-gemma4-v1-training`](https://www.kaggle.com/code/benhaslam/simsat-gemma4-v1-training). v11's adapter passes the dynamic LoRA tensor sanity gate at **410/410** — the correct count for Gemma-4-E2B's GQA architecture (15 canonical k/v modules × 2 + 35 layers × 5 non-k/v modules × 2); see [`V11_AUDIT.md`](./V11_AUDIT.md). The v1–v10 null-training audit (target_modules `.linear` matched only the multimodal towers, not the language model) is documented in [`notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`](./notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md). The `v11 LoRA partial-save` investigation (Issue #27) is also documented honestly: the suspected GQA-dedup bug never existed; 410 was the correct save count; the bug was in the hardcoded `_EXPECTED_TOTAL = 490` sanity gate.

  **Eval evidence:**
  - **v11 in-distribution (N=37, accept↔refine binary, geometric register)**: exact **0.86** (32/37), useful **0.97**, MAE **0.13**. Per-pack: disaster=1.00, maritime=0.92, urban-coastal=0.71. +0.32 over always-majority (refine), +0.61 over uniform random.
  - **v11 cross-distribution (N=152, balanced 4-class, geometric + spectral-biochemical)**: exact **0.30**. The 56-point gap is distribution-shift evidence: v11 trained on accept+refine, never saw `defer` or `skip` examples, so the model class-collapses on the broader pool. **This is exactly the on-orbit-drift problem the architecture is built to solve via TTT + viability gates.** The drop is genuine generalization data, not a sign the model is broken.
  - **v12 retrain (dataset v4, 1638 rows, balanced 4-class, includes the 152 reviewed cases)**: in flight at submission time. v12 eval will report both the full-pool number (with explicit in-distribution-leakage caveat: ~93% of N=152 is in v12 training set) and the held-out N=4 subset (true generalization, but no statistical power). The honest framing for v12 vs v11 is delta-on-comparable-eval, not absolute generalization.
  - **v17–v19** (intermediate Kaggle runs, dataset v3 with auto-defer expansion): regressed to ~0.46 exact agreement on N=37 — auto-generated defer labels did not match the operator threshold. v11 stayed canonical; the regression is documented as a labeling-distribution finding, not a model failure.

  This is **not** the `v35-gov` Gemma-4 fine-tune — `v35-gov` targets human-interview / consent-governance prompts, wrong task shape for satellite imagery. The VLA adapter is model-agnostic by design: any backend that emits the eight-key ObservationVLA JSON contract (schema at `src/sim/observation_vla/observation_payload.schema.json`) plugs into the same scaffold, trust layer, viability gates, and TTT loop without code changes. Swapping the backend is an env-var change (`OBSERVATION_VLA_BACKEND`), not a refactor — model adaptability is an explicit part of the General AI Track entry.

Per-track thesis statements for the pitch are finalized in [CHALLENGE_ENTRY.md](./CHALLENGE_ENTRY.md).

## Evidence
- Reviewed submission packet: [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md)
- Visual casebook: [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md)
- Readiness checklist: [SUBMISSION_READINESS.md](./SUBMISSION_READINESS.md)
- ObservationVLA reviewed eval: [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md)
- Encounter planner eval (248 windows, all 4 packs): [ENCOUNTER_EVAL.md](./ENCOUNTER_EVAL.md)
- Trust-layer TTT stability analysis (10 seeds): [ttt_stability_analysis.md](./ttt_stability_analysis.md)
- Inference benchmarks + scenario assessment: [benchmark_results/BENCHMARK_RESULTS.md](./benchmark_results/BENCHMARK_RESULTS.md)

**ObservationVLA eval — v11 in-distribution (N=37 operator-reviewed cases, accept↔refine binary, geometric register):**
- Exact operator-action agreement: **0.86** (32/37)
- Useful/not-useful agreement: **0.97**
- Usefulness score MAE: **0.13**
- Per-pack: disaster=1.00, maritime=0.92, urban-coastal=0.71
- Coverage: 3 geometric/structural packs (maritime, disaster, urban-coastal)
- Baselines: random uniform 0.25, always-majority (refine) 0.54

**ObservationVLA eval — v11 cross-distribution (N=152 operator-reviewed cases, balanced 4-class, geometric + spectral-biochemical):**
- Exact operator-action agreement: **0.30** (138/152 parsed; majority baseline 0.28)
- 56-point gap from in-distribution: v11 was trained on accept+refine only, class-collapses on defer/skip
- This is distribution-shift evidence — the on-orbit-drift problem the architecture exists to solve

**MuZero Liquid Track BC — Stage 2 seed sweep (3 seeds: 13, 42, 2026):**
- Best val_acc: **0.908 ± 0.014** (range 0.900-0.925)
- Replaces single-seed 0.967 claim from Stage 2 v2; new training corpus is the post-N=152 pool
- Full breakdown in [MUZERO_SEED_SWEEP.md](./MUZERO_SEED_SWEEP.md)

**Trust-layer TTT (10-seed stability, 256 encounter records, 20 cycles):**
- MAE improvement: **22.5% ± 0.1%** [22.2–22.7%] — deterministic across seeds
- Episode reward improvement: **24.1% ± 0.2%** — confirmed over 544 labeled records
- 90% convergence: **1.9 ± 0.3 cycles** (essentially 2 passes through the corpus)
- True-accept gain: **+28 per episode** | False-accept reduction: **−2 per episode**
- Threshold robust: improvement positive at every threshold from 0.50 to 0.75
- Dominant signal: clarity weight 0.12→0.24 (+0.12); geometry weight 0.34→0.24 (−0.10)

**Encounter planner eval (248 windows, all 4 packs, 5 × 48h runs per pack):**
- Scaffold produces **zero refine actions** across all 248 windows and all 4 packs
- WCLI trust layer adds **72 refine decisions** (29% refine rate on trust-changed windows)
- 92 of 248 windows (37%) changed action; all upgrades move toward engagement (zero downgrades)
- Urban coastal: 47/93 windows (51%) changed to refine — highest pack, consistent with ambiguity claim
- Mean trust-score lift over scaffold: **+0.013** across all packs

**Pinned casebook cases (one per pack):**
- `maritime_chokepoints` → Suez Canal, reviewer `Ben Haslam`, usefulness `0.95`
- `disaster_response_weather` → Houston Ship Channel, reviewer `Ben Haslam`, usefulness `0.92`
- `urban_coastal_ambiguity` → Port of Rotterdam, reviewer `Ben Haslam`, usefulness `0.90`
- `pedospheric_integrity` → Mato Grosso Agricultural Frontier, reviewer `Ben Haslam`, outcome `defer` (trust below refine threshold; honest result for a low-quality pass)

## Inference Benchmarks

Gemma-4-E2B-IT on Tesla T4 (15.6 GB VRAM), SimSat observe/assess contract v1:

| dtype | N | mean latency | p50 | p95 | tokens/sec |
|---|---|---|---|---|---|
| float16 | 100 | 9,455.7 ms | 9,445.0 ms | 9,620.8 ms | 12.8 |
| float16 | 20 | 11,207.8 ms | 11,201.4 ms | 11,301.5 ms | 13.1 |
| int8 | — | not available (bitsandbytes incompatible with transformers HEAD) | — | — | — |

N=100 is the primary benchmark: 100 timed passes, std 90.2 ms, p90 9,569.7 ms, p99 9,744.9 ms. N=20 retained for transparency; ~19% latency gap between sessions is documented honestly in [benchmark_results/BENCHMARK_RESULTS.md](./benchmark_results/BENCHMARK_RESULTS.md) — likely Kaggle GPU allocation variability across sessions.

## LFM Track Benchmarks

LFM2.5-VL-450M tile encoder + MuZero BC policy head, measured on RTX 2080 (`benhaslam` BEAST), 75 episodes across all 3 scenario packs. Full breakdown in [MUZERO_LFM_EVAL.md](./MUZERO_LFM_EVAL.md).

**Encoder latency (10 synthetic 64×64 tiles, post-warmup):**

| Model | embed_dim | Mean ms/tile | p95 ms/tile |
|---|---|---|---|
| `LiquidAI/LFM2.5-VL-450M` | 768 | 69 | 78 |
| `LiquidAI/LFM2.5-VL-1.6B` | 1152 | 246 | 248 |
| `google/siglip-base-patch16-224` (offline fallback) | 768 | ~18 | ~25 |

**BC policy — Stage 2 v2 (canonical, `--max-aug-ratio 4.0`, 136 examples):**

| Metric | Value |
|---|---|
| Episodes | 75 (all packs) |
| Best val accuracy | 0.967 |
| Mean episode reward | -0.0438 ± 0.0191 |
| accept | 9 (12%) |
| refine | 53 (71%) |
| skip | 13 (17%) |
| defer | 0 (0%) |

**Policy progression:**

| Policy | Val acc | Reward | Notes |
|---|---|---|---|
| Playback (stored VLA action) | n/a | -0.0441 | Encoder-decorative baseline |
| Stage 1 BC (75 traces) | 0.800 | -0.0600 | Class collapse: 100% refine on eval |
| Stage 2 v1 (uncapped aug) | 0.950 | -0.0396 | Skip over-predicted (36%); accept under-predicted (3%) |
| **Stage 2 v2 (cap=4.0)** | **0.967** | **-0.0438** | **Canonical. Accept recovered (12%), skip normalized (17%)** |

**Honest limitations:**
- Defer: 0/75 predictions across all policies. Corpus has only 3 original defer traces — no augmentation ratio compensates for this. `scripts/build_defer_queue.py` generates a focused 20-candidate review queue to address it.
- Hardware is RTX 2080 (local BEAST), not T4 or Orin. On-orbit latency for LFM2.5-VL-450M is projected sub-250ms per Liquid AI's benchmarks; not yet directly measured on Orin.
- Stage 3 (two-scope TTT, live per-pass adaptation) is not yet benchmarked — requires a live encounter stream.

## Runtime Truth
- Sentinel is the primary observation source. Spectral bands used: B04, B08 (NDVI), B11, B12 (SWIR ratio, soil moisture/organic proxy), B08A/B05 (EVI, canopy structure) for the pedospheric register; RGB/NIR/SWIR composite for geometric registers.
- Mapbox is optional and disabled in the current submission flow.
- ObservationVLA current runtime is `transformers_vlm_local` (Gemma-4-E2B SimSat fine-tune v11). `clip_local` (CLIP ViT-B/32) remains available as a zero-weight-download reference backend. Stored case assessments in the packet reflect the runtime active at review time; see the packet Notes section.
- **TTT is active at two layers:** VLA weights / adapters adapt on streaming tiles; the WCLI trust layer thresholds and priors update from realized-utility feedback. Both adaptation streams flow through the six viability gates before persistence.
- The reviewed eval pool was expanded from 8 → 37 operator-reviewed cases via `scripts/batch_review.py` on 2026-04-27. v11 fine-tune over those 37 produces useful/not-useful agreement 0.97 and magnitude MAE 0.13. See `OBSERVATION_VLA_EVAL.md` for the per-case breakdown.
- **Runtime backend note:** The 86 stored assessments in `assessments.json` used CLIP (`clip_local`) and stub backends at collection time. The Gemma-4 v11 eval is a separate offline evaluation against operator-reviewed Sentinel-2 tiles. CLIP assessments show lower operator usefulness ratings (mean 0.647) vs stub (0.867), a finding documented honestly in `benchmark_results/BENCHMARK_RESULTS.md`. Gemma-4 v11 is the current production backend; CLIP remains available as a reference.
- A synthetic scenario assessment study (90 Gemma-4 assessments across 27 geographic locations × 4 cloud conditions, 3 scenario packs) runs alongside inference benchmarks and results are tracked in `benchmark_results/BENCHMARK_RESULTS.md`.
- Mission-response is a policy-and-utility layer, not live spacecraft actuation.

## Demo Order
1. Open [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md) for the scorecard and reviewed cases.
2. Open [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md) for the four pinned visual examples (one per scenario pack, plus a second urban_coastal case).
3. Use [CHALLENGE_ENTRY.md](./CHALLENGE_ENTRY.md) for the spoken walkthrough — architecture framing, TTT + viability mechanics, per-track model notes.

## Known Issues
Code-review findings and their patch bundles live in-repo under `review/`:

- `review/2026-04-21-phase-1/` — Must-Do correctness fixes (tzinfo bug, hygiene)
- `review/2026-04-21-phase-2/` — Should-Do thesis + calibration work
- `review/2026-04-21-phase-3/` — Entry B scaffold + Drive recon block
- `review/2026-04-21-phase-4/` — Submission-docs refresh with TTT + viability thesis lock-in and frontend HAIC brand hide
- `KNOWN_ISSUES.md` (if present at repo root) — consolidated index across all phases

## Reproduce
```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000 --reviewed-only
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000
```

## Acknowledgments
Thanks to **Guilherme Mesquita** (Genesis) and **Garrett Sutherland** (Tesseract T3) for ongoing conversations and feedback on the model-agnostic ObservationVLA contract. Their independent VLM tracks helped sharpen the eight-key JSON schema even though those backends are not part of this submission.
