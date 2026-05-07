# SimSat Challenge Entry

## Thesis
SimSat is a governed on-orbit continual-learning loop. It reframes mission operations as a sequence of encounter windows, runs stacked test-time training at the vision-language and trust-layer levels, and gates every adaptation through six non-compensatory viability checks. The result is a pipeline that demonstrably improves with every pass instead of drifting, with no ground-truth validator required — which is why it has to run in orbit.

The system spans two structurally different observational registers: geometric and structural (maritime chokepoints, disaster response, urban coastal scenes) and spectral-biochemical (pedospheric integrity — soil health expressed through NDVI, SWIR ratio, and vegetation indices over active degradation sites). The same encounter planner, trust layer, viability gates, and TTT loop operate across both registers without architectural modification.

Two planners compete over the same ranked windows:
- `Scaffold`: deterministic geometry-and-availability scoring
- `WCLI Trust`: scaffold plus a trust-gated support layer that can `accept`, `defer`, `skip`, or `refine`

For the shortest reviewer-facing summary of the current state, use [SUBMISSION_BRIEF.md](./SUBMISSION_BRIEF.md).

The competitive claim is simple and testable:

> Treating observation opportunities as structured encounter episodes, and allowing both the VLA backend and the trust gate to adapt at test time under a non-compensatory viability filter, reduces wasted high-cost materialization while preserving high-value windows — and does so without a ground-truth validator, which is why it must run in orbit.

This challenge entry is intentionally `no-Mapbox-safe`: the core planner, evaluation loop, ObservationVLA trace pipeline, and TTT layers do not require a Mapbox token.

## Why this must run in orbit
Three stacked constraints force the full pipeline to the spacecraft, in priority order.

### 1. Distribution shift without a ground-truth validator
The differentiator. A satellite streaming imagery hits real distribution drift — new geographies, seasonal variation, cloud and sensor conditions — with no path back to ground for label collection at any useful cadence. Adapting a model to that drift without ground-truth labels is the open on-orbit continual-learning problem. Our answer is six non-compensatory viability gates that every candidate observation must pass before it is allowed to update system state:

1. **Information gain** — The adaptation must measurably reduce the model's uncertainty over the current observation distribution. Observations that confirm what the model already predicts are operationally useful but generate no adaptation signal — the encounter was informative, but it taught the model nothing new.
2. **Observation quality** — The tile must clear minimum acquisition thresholds: cloud cover below the scene-specific ceiling, target visible, sensor geometry within valid range. Partially or fully occluded passes do not generate adaptation signal regardless of predicted class.
3. **Metadata consistency** — Claimed metadata (cloud fraction, sensor band, orbit timestamp, target position) must match measured image properties. A claimed clear-sky tile that actually shows 85% cloud cover is rejected outright; adapting on false positives would corrupt the model's calibration.
4. **Update magnitude** — The weight delta from any single pass is bounded. An update that shifts the policy by more than 30% from its pre-pass baseline is treated as an artifact or sensor anomaly, not genuine drift, and is blocked before persistence.
5. **Update rate** — Adaptations are rate-limited across passes. Rapid consecutive updates on the same target type cause local overfitting; the gate enforces a cumulative count ceiling and suppresses adaptation when frequency outpaces what the observation stream can genuinely support.
6. **Error balance** — If recent adaptations have consistently pushed the policy in the same direction, the gate flags systematic bias and suppresses further updates until a correcting signal arrives. Reinforcing a systematic error is strictly worse than not adapting.

These six gates convert a naively dangerous "let the model learn from what it sees" loop into a governed one. Without them, on-orbit TTT is uncontrolled drift; with them, every accepted update carries quality-verified, magnitude-bounded, bias-checked information about the world before it touches model state.

**Selective downlink.** Observations that clear all six gates *and* exceed the high-confidence accept threshold (trust score ≥ τ_downlink) are candidates for selective ground transmission — imagery and derived assessments packaged for the highest-value passes only. Below that threshold, processed results stay on-orbit. The satellite sends pixels to the ground only when they are genuinely worth the downlink budget.

### 2. Bandwidth
The hackathon's prize-compute budget allocates 5 MB uplink and 10 MB downlink per satellite-month. That envelope cannot ferry model weight updates to ground and back at any useful cadence. Any TTT that is going to track drift must happen on the satellite itself.

### 3. Latency
The next encounter window arrives in minutes, not hours. Adaptations discovered in this pass have to be usable in the very next pass — there is no round-trip budget for ground re-training and re-upload.

## Test-Time Training — Stacked
The entry runs TTT at two layers simultaneously. Both adaptation streams flow through the viability gates above; nothing persists that fails any gate.

### VLA-layer TTT
The vision-language backend — LFM2.5 in the Liquid Track, Gemma-4 in the General AI Track — adapts on streaming Sentinel tiles. Lightweight adapters or weight updates nudge in response to each incoming observation that passes the viability gates. Over a pass, the backend drifts toward the actual data distribution the satellite is seeing right now rather than the distribution its training set was frozen on.

### Trust-layer TTT
The WCLI gate tunes itself online from realized-utility feedback. When the planner commits to a `materialize_now` action and the downstream observation resolves as useful-or-not, that signal updates the trust gate's thresholds and priors. Over time the trust layer learns which reasons justify an `accept` for *this* orbit, *this* season, *this* sensor configuration — not generic population-level reasons.

### Viability gate as the common filter
Every candidate adaptation — whether a VLA-layer weight update or a trust-layer threshold change — passes through the six-gate filter before persistence. The architectural claim: stacked TTT is only safe on-orbit because the gates screen out bad updates autonomously. There is no human in the loop during operations; the gates are the substitute.

Gates 1–3 apply at observation intake: quality and metadata checks happen before any adaptation signal is generated. Gates 4–6 apply to the adaptation response itself and are implemented in `evaluate_ttt_viability()` (`src/sim/haic/viability.py`): update magnitude (weight_drift ≤ 0.30 from pre-pass baseline), update rate (cumulative count ceiling), and error balance (<70% same-sign errors across the last N updates).

The gates operationalize a formal viability condition: a learning system remains stable when `C_eff(t) ≥ E(t)`, where E(t) is the expected per-iteration divergence from the external world and C_eff(t) is the corrective capacity of independent grounding signal. Gates 1–3 ensure the incoming signal is real, quality-verified, and physically consistent — a necessary condition for C_eff > 0. Gates 4–6 ensure the adaptation response is proportionate and not compounding prior errors — necessary to keep E(t) bounded. All six must pass; failure in any single gate blocks the entire adaptation. Without the gates, on-orbit TTT accumulates uncorrected drift by construction. With them, every accepted update must clear the full condition before touching model state.

### What is proven vs. what the prize hardware enables

Three drift-handling mechanisms in the architecture, three different states of evidence:

| Mechanism | Status in this submission | Evidence |
|---|---|---|
| **Trust-layer TTT** | **Proven in flight (offline replay)** | 10-seed analysis: **22.5% MAE improvement ± 0.1%**, 24.1% reward improvement, 90% convergence in 1.9 ± 0.3 cycles. The `accept→refine` calibration loop on the Rotterdam pinned case (`urban_coastal_ambiguity`, trace `4f65355f…`) is a single-trace illustration of the same mechanism: WCLI says refine on 49% cloud compound risk; operator says accept on visible-portion sufficiency; the override becomes signal for the trust layer's online update. See [`ttt_stability_analysis.md`](./ttt_stability_analysis.md). |
| **Six viability gates** | **Implemented and exercised** | `evaluate_ttt_viability()` in `src/sim/haic/viability.py`, wired via `_apply_trust_layer_ttt()` in `service.py`. Fired correctly during the 2026-05-06 operator-review session — `weight_drift` and `error_bias` warnings logged in `batch_review.py` output as the trust layer's thresholds drifted under the new label distribution. The gates are gate-able evidence that the safety mechanism activates under realistic load. |
| **VLA-layer TTT** | **Wired, requires live encounter stream** | Stage 3 two-scope TTT is wired in both tracks (`src/sim/muzero/`, `src/sim/observation_vla/`). Not yet benchmarked end-to-end on a live stream — the 75-episode MuZero offline-replay eval and the 152-case ObservationVLA eval are both on **static** checkpoints. This is the lane the prize hardware (NVIDIA Orin 16 GB) opens. |

The ObservationVLA fine-tunes — v11 (canonical) and v12 (retrain in flight on dataset v4) — are **static checkpoints that set where drift starts**, not drift-adaptive models in their own right. The 56-point gap between v11's in-distribution N=37 (0.86) and cross-distribution N=152 (0.30) eval frames is **what TTT is meant to close per-pass on a live stream**, not what offline retraining closes. v12's contribution is to move the static training distribution closer to deployment so the live TTT loop covers less ground per pass — a complement to the runtime mechanism, not a replacement.

What the prize buys: live Orin time to run Stage 3 VLA-TTT against an actual encounter stream and benchmark per-pass adaptation convergence under the same six viability gates that already govern the trust layer in offline replay.

## Observational Scope — Two Registers

The four scenario packs span two structurally different signal types.

The first three packs operate in the **geometric register**. The signal of interest is structural: ship positions at a chokepoint, storm-damage extent along a coastline, port-complex geometry at a dense logistics hub. The trust layer learns when geometric confidence justifies commit versus refine or defer; the VLA backend learns to distinguish genuine structure from clutter, perspective ambiguity, and sensor noise.

The `pedospheric_integrity` pack operates in the **spectral-biochemical register**. The signal of interest is soil health expressed through Sentinel-2's multi-spectral bands:

- **NDVI** `(B08 − B04) / (B08 + B04)` — vegetation density and stress. Declining NDVI over irrigated farmland signals salinization or waterlogging before visible bare-soil emergence.
- **SWIR ratio** `B11 / B12` — clay content, soil moisture, and organic matter proxy. Rising SWIR ratio tracks the loss of soil carbon and moisture retention as forest-to-agriculture conversion advances.
- **EVI** — canopy structure indicator, less susceptible to saturation than NDVI in high-biomass scenes, sensitive to early-stage canopy thinning at deforestation boundaries.

Degradation at the three target sites manifests as spectral shift rather than geometric pattern: salinization at the Nile Delta pushes NDVI down while SWIR ratio rises; active deforestation at the Mato Grosso boundary produces sharp EVI and NDVI edges that advance between passes; groundwater depletion at Punjab compresses the seasonal NDVI amplitude over successive crop cycles. The accept/refine/defer pressure comes from agricultural cycle timing, monsoon and wet-season cloud cover, and the difficulty of distinguishing chronic degradation from reversible seasonal stress in multi-pass temporal sequences — structurally different from the geometry-driven ambiguity in the other three packs.

This distinction matters architecturally. A system that works only in the geometric register could succeed through structural heuristics without genuine image-conditioned spectral reasoning. A system that handles both registers — under the same encounter planner, trust layer, viability gates, and TTT loop — demonstrates that the governed continual-learning architecture generalizes across observational domains. That is the claim the pedospheric pack tests.

## Two-Track Submission
This repo is submitted to both tracks of the AI in Space hackathon. The scaffold, WCLI trust layer, mission-response layer, propagator, viability gates, and TTT infrastructure are identical across tracks — only the VLA backend changes.

- **Liquid Track (LFM2.5 + MuZero)** — A vision-language model serves as the Sentinel tile encoder: it processes each raw tile into a dense `(embed_dim,)` embedding vector that feeds MuZero's `h()` representation function. MuZero + MCTS then plans over encounter windows using that embedding. The hybrid is deliberate: MuZero's ResNet weighs a few MB and fits the satellite's 5 MB uplink budget; the encoder is LoRA fine-tuned on SimSat Sentinel tiles and runs locally for its single encoder pass per window. Training follows the ARC3 three-stage chain — (1) pretrain on the 86-trace Sentinel tile corpus, (2) fine-tune with SimSat scenario-pack augmentation, (3) two-scope TTT per live pass using the `TTTScope1` / `TTTScope2` budget schedules ported from `arc3_game.py`. The MuZero game adapter (`SimSatGame`), gym environment (`SimSatEnv`), FC-network config (`SimSatMuZeroConfigLFM`), and encoder protocol (`TileEncoder` / `build_encoder("lfm2vl")`) are all in `src/sim/muzero/`. The encoder slot is filled by **[`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)** (450M params total, vision tower is SigLIP-2 NaFlex shape-optimized 86M with 768-dim pooled output, sub-250ms edge inference per Liquid AI's benchmarks); the [1.6B variant](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) is a one-line `model_id` swap if a larger encoder is preferred. SigLIP-base remains in the codebase as the offline-safe `HFVisionTowerEncoder()` default. **Liquid Track eval** (`scripts/muzero_lfm_eval.py`, 75 matched Sentinel tiles, all 3 scenario packs): with **LFM2.5-VL-450M** as the encoder (the default since 2026-04-27): encoder latency **67.9 ms/tile avg** (p95 69.3 ms) on RTX 2080; embed_dim=768; action distribution accept=17%, refine=68%, defer=5%, skip=9%. The 1.6B variant runs at 246 ms p95 (matches Liquid AI's "sub-250ms edge inference" claim). **Stage 1 pretrain** (`scripts/muzero_stage1_pretrain.py`) reaches val_acc **0.800** but argmax-collapses to refine in deployment due to a 64% refine corpus — the lesson that motivates Stage 2. **Stage 2 pretrain** (`scripts/muzero_stage2_pretrain.py`, scenario-pack augmentation: flips/rotations/jitter with per-class augmentation cap to prevent over-amplifying minority originals) reaches val_acc **0.967** on a 28-sample stratified val (accept 10/10, defer 3/3, refine 10/10, skip 6/7). Wired into `muzero_lfm_eval.py --policy bc --policy-head weights/muzero/stage2_bc/policy_head.pt`, the Stage 2 head produces diverse predictions (accept 12%, refine 71%, skip 17%) closely matching the operator distribution, with mean episode reward **−0.0438** (within noise of playback's −0.0441 and well above Stage 1's −0.0600). Defer was 0/75 in Stage 2 v2. Corpus expanded to 13 defer cases (2026-05-05); Stage 2 v3 confirmed the boundary is learnable (0→32/75 defer in eval) but over-predicts defer (43% vs 5% operator baseline) — auto-labeled corpus is too concentrated. v2 remains canonical; v3 is a research checkpoint documented in [MUZERO_LFM_EVAL.md](./MUZERO_LFM_EVAL.md). Full per-pack breakdown + variant comparison + Stage 1/2 details in [MUZERO_LFM_EVAL.md](./MUZERO_LFM_EVAL.md).
- **General AI Track (Gemma-4)** — ObservationVLA backed by a Gemma-4-E2B fine-tune scoped specifically to SimSat's encounter-triage task (accept / defer / skip / refine over Sentinel-style tiles). Training code and dataset at `notebooks/kaggle-simsat-gemma4-v1/`; Kaggle kernel `benhaslam/simsat-gemma4-v1-training` (**v11** = first run with corrected target_modules — see [`GEMMA4_LORA_NULL_TRAINING_AUDIT.md`](./notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md) for why v1-v10 trained nothing); v11 trained on dataset v2 (713 ChatML rows, accept+refine dominant); v12 retrain in flight on dataset v4 (1638 rows, balanced 4-class, includes 152 fresh operator-reviewed labels from gallery_review.html). Two complementary eval frames:
  - **v11 in-distribution (N=37 accept+refine, geometric register)**: exact 0.86 / useful 0.97 / MAE 0.13. Per-pack disaster=1.00, maritime=0.92, urban-coastal=0.71.
  - **v11 cross-distribution (N=152 balanced 4-class, includes spectral-biochemical pedospheric)**: exact 0.30 — class collapse on out-of-distribution defer/skip cases. The 56-point delta is distribution-shift evidence, the architecture's central claim.
  - **v12 (in flight)**: retrained on the broader 4-class corpus; eval will report both full-pool (in-distribution leakage acknowledged: ~93% of N=152 is in training set) and held-out N=4 subsets to keep the comparison honest.

  See [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md) for the full per-case table, baselines, and caveats. This is **not** the `v35-gov` Gemma-4 fine-tune — `v35-gov` targets human-interview and consent-governance prompts, wrong task shape for satellite imagery. The VLA adapter is model-agnostic by design (`src/sim/observation_vla/transformers_vlm_local.py`): any backend that emits the eight-key ObservationVLA JSON contract (schema at `src/sim/observation_vla/observation_payload.schema.json`) plugs into the same scaffold, trust layer, viability gates, and TTT loop without code changes. Swapping backends is an env-var change (`OBSERVATION_VLA_BACKEND`), not a refactor. The competitive claim extends to this: the SimSat pipeline is model-agnostic by design — the governed continual-learning architecture is the contribution, not any one model weight checkpoint. See `COLLABORATOR_GUIDE.md` for the integration contract.

**Per-track pitch thesis (one sentence each):**

- **Liquid Track:** SimSat couples a vision-language tile encoder (SigLIP-base today, LFM2.5-VL the moment Liquid AI publishes weights — same one-line slot) with a MuZero planner — the hybrid fits a 5 MB satellite uplink budget while two-scope test-time training adapts both the encoder and the planner in-pass without ground-contact.
- **General AI Track:** SimSat fine-tunes Gemma-4-E2B specifically for satellite encounter triage and wraps it in a model-agnostic scaffold with viability-gated test-time training, so any vision-language model that emits the eight-key ObservationVLA JSON contract inherits the same governed continual-learning loop with zero code changes.

## What Is New
- Real future encounter windows from the live TLE-backed propagator
- Scenario packs for challenge evaluation spanning **two observational registers**: geometric/structural (maritime chokepoints, disaster response, urban coastal) and spectral-biochemical (pedospheric integrity)
- **Pedospheric integrity monitoring** via NDVI, SWIR ratio (B11/B12), and EVI over three active degradation sites — Nile Delta salinization, Mato Grosso deforestation boundary, Punjab groundwater depletion — demonstrating the same architecture operates across structurally different signal types without modification
- Decision-level deltas between scaffold and WCLI-trust planners
- Materialization-yield comparison across the same ranked windows
- ObservationVLA lane with image-conditioned assessment, labeled trace capture, and lightweight pass/scenario/mission calibration hooks
- **Stacked test-time training** at the VLA and trust-layer levels
- **Six non-compensatory viability gates** screening every candidate adaptation before it updates system state
- Explicit runtime capability reporting so local runs can verify that Mapbox is optional
- Mission-response layer that turns effective observation decisions into concrete downstream actions and simulated utility

The current ObservationVLA backend evaluation is tracked separately in [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md).

Core implementation lives in:
- [service.py](./src/sim/encounter/service.py)
- [planner.py](./src/sim/encounter/planner.py)
- [trust_model.py](./src/sim/encounter/trust_model.py)
- [targets.json](./src/sim/data/encounter/targets.json)

## Scenario Packs
- `maritime_chokepoints`: Suez, Panama, Singapore
- `disaster_response_weather`: flood/storm-sensitive coastal and industrial targets
- `urban_coastal_ambiguity`: dense mixed-use port and shoreline scenes where tempting windows may still deserve refinement
- `pedospheric_integrity`: soil-health and land-degradation monitoring — Nile Delta salinization, Mato Grosso deforestation frontier, Punjab high-intensity agriculture

The first three packs exercise accept/refine/defer decisions under geometric and cloud ambiguity in maritime and urban contexts. The `pedospheric_integrity` pack tests the same planner and trust layer in a different observational register: the signal of interest is soil health expressed through NDVI, SWIR, and Red Edge spectral bands rather than structural geometry. Seasonal cloud cover (Mato Grosso wet season, Punjab monsoon) and agricultural cycle timing (Nile Delta) create genuine defer and refine pressure that is structurally different from the geometry-driven ambiguity in the other packs. Adding this pack validates that the architecture generalises across observational domains without modification — the same encounter planner, trust layer, viability gates, and TTT loop operate identically regardless of what the satellite is looking at or why.

These packs are encoded directly in [targets.json](./src/sim/data/encounter/targets.json) for reproducibility.

## Evaluation Outputs
Each evaluation logs:
- action distributions for scaffold and WCLI-trust planners
- action transition counts such as `accept->refine`
- materialization yield for top-ranked candidates
- decision-level deltas showing which windows changed, by how much, and why

The most challenge-relevant artifact is the delta list: it makes the WCLI trust layer falsifiable instead of rhetorical.

**Encounter planner eval summary** (248 windows across all 4 packs, 5 × 48h runs per pack — see [ENCOUNTER_EVAL.md](./ENCOUNTER_EVAL.md)):

The scaffold planner produces **zero refine actions** across all 248 windows. Every refine decision in the system originates from the WCLI trust layer detecting compound risk the deterministic scaffold cannot express. WCLI trust added 72 refine decisions (29% of trust-changed windows) and changed the action on 92 of 248 windows (37%) — all upgrades, zero downgrades.

| Pack | Windows | Trust refine | Trust score lift |
|------|---------|--------------|-----------------|
| maritime\_chokepoints | 50 | 10 (20%) | +0.014 |
| disaster\_response\_weather | 50 | 10 (20%) | +0.014 |
| urban\_coastal\_ambiguity | 93 | 47 (51%) | +0.013 |
| pedospheric\_integrity | 55 | 5 (9%) | +0.012 |
| **All packs** | **248** | **72 (29%)** | **+0.013** |

**Trust-layer TTT results** (10-seed stability, 256 records, 20 cycles — see [ttt_stability_analysis.md](./ttt_stability_analysis.md)):

| Metric | Value |
|--------|-------|
| MAE improvement | **22.5% ± 0.1%** (10 seeds, range 22.2–22.7%) |
| Episode reward improvement | **24.1% ± 0.2%** (SimSatEnv, 544 labeled records) |
| 90% convergence | **1.9 ± 0.3 cycles** |
| True-accept gain | +28 per episode (116 → 144) |
| False-accept reduction | −2 per episode (4 → 2) |
| Threshold robust | Positive improvement at all thresholds 0.50–0.75 |

The improvement is essentially deterministic (std < 0.2% across 10 random seeds). Convergence in under 2 full passes through the corpus is consistent with the on-orbit constraint: the system reaches near-optimal adapted weights before the satellite completes its first full target revisit cycle.

## ObservationVLA Reality Check
The ObservationVLA lane is backed by the Gemma-4-E2B SimSat fine-tune **v11** (`OBSERVATION_VLA_BACKEND=gemma4`). The current reviewed evaluation:

- reviewed set: **N=37** (operator_reviewed traces; pool grew via batch_review.py session 2026-04-27)
- exact action agreement: **0.86** (32 of 37 traces match the operator's recommended action)
- bucketed action agreement: **0.86**
- useful / not-useful agreement: **0.97**
- usefulness-score MAE: **0.13**
- the model should be treated as a calibrated evidence lane, not a trusted autonomous action policy
- the `clip_local` CLIP baseline remains available as a zero-weight-download reference

**Version note (2026-04-27).** v1 through v10 of the SimSat fine-tune all
shared a `target_modules` config bug — the `.linear` suffix matched only
Gemma-4's vision/audio towers, not the language model decoder layers. Those
adapters trained zero language-model parameters; every prior eval number
described stock `google/gemma-4-E2B-it`, not a fine-tune. The bug, the
audit, and the corrected regex are documented in
[`notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`](./notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md).
v11 is the first run with actual language-model LoRA training (410 LoRA
tensors, 100% on `language_model`, all `lora_B` non-zero), and a post-save
sanity gate now fails the kernel loudly if a future run regresses to a
no-op adapter. Prior numbers (MAE 0.16, bucketed 0.57 on N=7) measured
the base model and overstated the fine-tune's contribution.

For the latest reviewed check:

```bash
python scripts/observation_vla_eval.py --inprocess
```

That writes [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md) with the current runtime, reviewed sample size, and conservative claim text.

## MuZero/LFM Track Reality Check

The MuZero BC policy (Stage 2 v2, canonical) is evaluated against 75 matched Sentinel tiles across all three geometric scenario packs. Encoder is `LiquidAI/LFM2.5-VL-450M` (768-dim SigLIP-2 NaFlex pooled output). Hardware: RTX 2080 (local). Full breakdown in [MUZERO_LFM_EVAL.md](./MUZERO_LFM_EVAL.md).

**Encoder latency (post-warmup, 10 synthetic 64×64 tiles):**

| Model | embed_dim | Mean ms/tile | p95 ms/tile |
|---|---|---|---|
| `LFM2.5-VL-450M` | 768 | 69 | 78 |
| `LFM2.5-VL-1.6B` | 1152 | 246 | 248 |
| `siglip-base-patch16-224` (offline fallback) | 768 | ~18 | ~25 |

**BC policy progression:**

| Policy | Val acc | Mean reward | Notes |
|---|---|---|---|
| Playback (stored VLA action) | n/a | −0.0441 | Encoder-decorative baseline |
| Stage 1 BC (75 traces) | 0.800 | −0.0600 | Class collapse: 100% refine on eval |
| Stage 2 v1 (uncapped aug) | 0.950 | −0.0396 | Skip over-predicted (36%); accept under-predicted (3%) |
| **Stage 2 v2 (cap=4.0)** | **0.967** | **−0.0438** | **Canonical. Accept 12%, refine 71%, skip 17%** |

Stage 2 v2 reward (−0.0438) is within noise of playback (−0.0441) and substantially better than Stage 1 (−0.0600), confirming the augmented BC head learns the operator distribution without overfitting the corpus skew.

**Honest limitations (disclosed, not hidden):**
- Defer: 0/75 predictions across all policies. Only 3 original defer traces exist — no augmentation ratio compensates for a corpus this thin. `scripts/build_defer_queue.py` generates a focused 20-candidate review queue; 10 operator-reviewed defer cases would enable Stage 2 v3.
- Hardware gap: encoder latency measured on RTX 2080, not NVIDIA Orin. On-orbit latency for LFM2.5-VL-450M is projected sub-250ms per Liquid AI's benchmarks; not yet directly measured on Orin.
- Stage 3 (two-scope TTT, live per-pass adaptation) is architecturally wired (`TTTScope1` / `TTTScope2` budget schedules, ported from `arc3_game.py`) but not yet benchmarked — requires a live encounter stream.
- The 75-episode eval uses matched Sentinel tiles across geometric packs only; `pedospheric_integrity` spectral-band tiles are not yet in the MuZero eval corpus.

## Judge-Facing Scorecard
For a compact, reproducible scorecard that can drop straight into notes or a submission draft:

```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000 --scenario-sweep --top-k 8 --materialize-top-k 2 --markdown
```

This prints one row per scenario pack with:
- candidate window count
- scaffold vs trust `accept` counts
- trust `refine` count
- number of changed decisions
- `accept->refine` transitions
- scaffold vs trust materialization yield
- the top trust/refinement reason

That scorecard is the quickest way to show operational benefit without a long live demo.

## Submission Evidence
For a concrete scenario-by-scenario evidence report built from the latest evaluation plus any labelled ObservationVLA traces:

```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000
```

This generates [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md) with one section per scenario pack containing:
- latest planner transitions and yield summary
- top delta highlights from the latest evaluation
- one curated case summary from a labelled trace, observation outcome, and mission-response action

The generated packet is Sentinel-first and prefers operator-reviewed labels when they exist. If no reviewed label exists yet, it falls back to `simulated_submission_case`.

To replace a simulated label with a real operator-reviewed outcome:

```bash
python scripts/review_queue_casebook.py --inprocess
python scripts/operator_review.py --base-url http://127.0.0.1:8000 --scenario-pack maritime_chokepoints
```

The review queue writes [REVIEW_QUEUE.md](./REVIEW_QUEUE.md) plus per-case images so the next human-review pass can compare the stored trace assessment with the current `clip_local` backend recommendation on the same imagery.

For a specific trace, inspect the full review bundle first:

```bash
python scripts/operator_review.py --base-url http://127.0.0.1:8000 --trace-id <trace_id> --show-bundle-only
```

Then replace the current label and pin it as the canonical submission case for that scenario:

```bash
python scripts/operator_review.py --base-url http://127.0.0.1:8000 --trace-id <trace_id> --reviewer "Your Name" --operator-action accept --useful true --usefulness-score 0.95 --pin-submission-case --pinned-by "Your Name"
```

Once one reviewed case is pinned per scenario pack, generate a strict reviewed-only packet:

```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000 --reviewed-only
```

That mode fails loudly if any scenario pack still lacks a pinned operator-reviewed submission case.

For a visual companion built from those pinned cases and their stored images:

```bash
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000
```

That generates [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md) plus local image assets for the pinned cases.

For a low-compute readiness check across the packet, casebook, pinned traces, and runtime honesty boundary:

```bash
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000
```

## Reproducible Demo
1. Start SimSat:
```bash
cd src/sim
python main.py
```
2. Run the challenge demo:
```bash
python scripts/challenge_demo.py --base-url http://127.0.0.1:8000 --scenario-pack maritime_chokepoints
```
3. Compare planners directly:
```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000 --scenario-pack disaster_response_weather --top-k 10
```
4. Open the dashboard and use the Encounter Planner panel to switch scenario packs, run evaluation, and inspect the top decision deltas.

You do not need `MAPBOX_ACCESS_TOKEN` for this flow. If Mapbox is disabled, the runtime exposes that fact through `/capabilities`, and the challenge scripts continue using Sentinel plus geometry support.

## Low-Compute Rehearsal
If local GPU and CPU are busy, stick to the evaluation route rather than a long interactive sim session:

```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000 --scenario-pack maritime_chokepoints --top-k 8 --materialize-top-k 2
```

That path exercises the challenge thesis with minimal load and still gives you transition counts, yield, and top changed decisions.

It is also the recommended path if you do not want to enable Mapbox billing at all.

## Suggested Live Demo Flow
1. Pick `maritime_chokepoints` and run evaluation.
2. Show shared windows and the scaffold vs WCLI-trust transition counts.
3. Highlight at least one `accept -> refine` delta and explain the trust reason.
4. Switch to `disaster_response_weather` and show that cloud or limited imagery support raises refine pressure.
5. Materialize one high-confidence `accept` and one trust-gated `refine` candidate to illustrate why the refine path exists.
6. Switch to `pedospheric_integrity` and show the spectral register: Sentinel B08/B04 (NDVI), B11/B12 (SWIR ratio) driving defer and refine pressure at the Mato Grosso and Nile Delta targets. The same encounter planner, trust layer, and viability gates handle this without modification — the architecture generalises across registers.
7. Close on the architectural claim: the same pipeline is running stacked TTT under six non-compensatory viability gates — the minimum safe configuration for on-orbit continual learning without a ground-truth validator.

## Known Issues and Review Bundles
The full in-repo code review from 2026-04-21 is preserved under `review/`:

- `review/2026-04-21-phase-1/` — Tier-1 correctness fixes (tzinfo bug at `src/sim/simulator.py:111`, hygiene cleanup, image reference fixes). **Apply these first.**
- `review/2026-04-21-phase-2/` — Tier-2 thesis/calibration work (accept→refine sweep, MAE calibration reframe, casebook auto-gen patch, architecture subsection).
- `review/2026-04-21-phase-3/` — Entry B backend scaffold (`entry-b/backend` branch; model-agnostic VLM adapter) and the Google Drive recon block.
- `review/2026-04-21-phase-4/` — This submission-doc refresh, TTT + viability thesis lock-in, and frontend brand-hide patch.

Known open gaps (disclosed, not hidden):

**Both tracks:**
- Stacked TTT is described architecturally. Live on-orbit demonstration requires the hackathon prize hardware (NVIDIA Orin 16GB); local demonstration shows the wiring and gating logic but not a full on-orbit drift trajectory.
- Stage 3 two-scope TTT (per-pass adaptation) is wired in both tracks but not yet benchmarked — requires a live encounter stream.

**General AI Track (Gemma-4):**
- The `accept→refine` thesis is demonstrated on the Port of Rotterdam pinned case (`urban_coastal_ambiguity`, `trace_4f65355f5c954fbf8db3fc684bb377af`) — scaffold `accept`, trust `refine`, operator confirmed `refine`, driven by cloud risk (48.78% cover).
- Accept-bias in the Gemma-4 fine-tune: Rotterdam refine cases still over-predicted as `accept`. Additional training epochs (v10 = 4 epochs) did not change this — requires dataset-level rebalancing.
- Fort Myers Coast miss: model predicts `defer` (score 0.20), operator says `refine` (score 0.90) — large abs_error on borderline partial-cloud windows.
- v11 "LoRA partial-save bug" investigation (Issue #27): a 410-vs-490 tensor count mismatch was initially diagnosed as a GQA-deduplication save bug. Investigation across v12–v19 revealed the opposite — **410 is the correct count** for Gemma-4-E2B's GQA architecture (15 canonical k/v modules × 2 + 35 layers × 5 non-k/v modules × 2). The bug was in the hardcoded `_EXPECTED_TOTAL = 490` sanity gate, not in the save logic. Fix in v19: dynamic `_EXPECTED_TOTAL = sum(...named_parameters() if "lora_A" in name or "lora_B" in name)`. v11 audited retroactively against the dynamic gate: **410/410 PASS**. v11's adapter is complete, on Hugging Face at [`HumanAIConvention/simsat-gemma4-v11`](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11). See [`V11_AUDIT.md`](./V11_AUDIT.md). v11 remains canonical for the headline 0.86 eval; v17–v19 retrained on dataset v3 (defer-expanded) and regress to ~0.46 — that is a labeling-distribution finding, not a save-logic finding.
- The ObservationVLA backend is model-agnostic by design — additional model adapters can be added by implementing the eight-key JSON contract in `src/sim/observation_vla/`. The current submission ships v11 only; alternative backends are not part of the entry.
- Large Sentinel tiles (full-resolution multi-spectral composites) caused the VLM processor to hang before inference. Fixed by capping image input to 448px before the processor call (`OBSERVATION_VLM_MAX_IMAGE_SIZE` env var, default 448). The pedospheric pinned trace was reviewed under the simulated backend prior to this fix; the resize path is exercised on any fresh encounter evaluation with the gemma4 backend active.

**Liquid Track (LFM2.5 + MuZero):**
- Defer class: 3 original traces → **13** after 10 auto-labeled defer outcomes added 2026-05-05 (cloud_cover≥80% AND target_visible heuristic). Stage 2 v3 (2026-05-05): defer went **0→32/75** in eval — the boundary is learnable. However v3 over-predicts defer (43% vs 5% baseline) because the auto-labeled corpus is too concentrated; v2 remains the canonical checkpoint. Full breakdown in [MUZERO_LFM_EVAL.md](./MUZERO_LFM_EVAL.md).
- Encoder latency measured on RTX 2080; on-orbit projection sub-250ms from Liquid AI benchmarks, but not yet directly measured on NVIDIA Orin.
- `pedospheric_integrity` spectral-band tiles not yet in the MuZero eval corpus — the 75-episode eval covers geometric packs only. Adding spectral register tiles is the next eval expansion.
- MuZero MCTS planning over encounter windows is implemented and wired; full tree-search planning is tested in simulation but not yet exercised in the live encounter planner loop.

## Acknowledgments
Thanks to **Guilherme Mesquita** ([Genesis](https://huggingface.co/HumanAIConvention/gemma4-haic-grounding-v1) and the orchOSModel project) and **Garrett Sutherland** (Tesseract T3) for collegial conversations and prompt-shape feedback as the ObservationVLA eight-key contract evolved. Their independent VLM tracks helped tighten the schema and the model-agnostic adapter pattern. Those backends are not part of this submission, but the collaboration improved it.

## Submission Framing
Use this wording in the pitch:

> SimSat is a governed on-orbit continual-learning loop. We model mission operations as a sequence of encounter windows, rank them with a cheap deterministic scaffold, and gate every decision through a WCLI-style trust layer that can refine instead of commit. A vision-language backend scores candidate tiles; a mission-response layer translates decisions into logged utility. On top of that, test-time training runs at two layers — the VLA adapts on streaming Sentinel tiles, and the trust gate tunes from realized-utility feedback — with both adaptation streams passing through six non-compensatory viability gates before anything persists. That gating mechanism is why this can run in orbit without a ground-truth validator, which is why it has to run in orbit.
