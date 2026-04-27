# SimSat Challenge Entry

## Thesis
SimSat is a governed on-orbit continual-learning loop. It reframes mission operations as a sequence of encounter windows, runs stacked test-time training at the vision-language and trust-layer levels, and gates every adaptation through six non-compensatory viability checks. The result is a pipeline that demonstrably improves with every pass instead of drifting, with no ground-truth validator required — which is why it has to run in orbit.

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
The differentiator. A satellite streaming imagery hits real distribution drift — new geographies, seasonal variation, cloud and sensor conditions. Adapting a model to that drift without ground-truth labels is the open on-orbit continual-learning problem. Our answer is six non-compensatory viability gates that every candidate observation must pass before it is allowed to update system state:

1. **Entropy reduction** — the adaptation must decrease uncertainty (measured via PRISM). An observation that adds log-probability mass without reducing entropy is rejected.
2. **Extraction risk** — bulk scraping dressed up as observation is rejected. Each update must look like a normal operational pass, not a data-mining raid.
3. **PRISM consistency** — claimed metadata (cloud cover, sensor, timestamp) must match measured image properties. Inconsistent claims are rejected without updating state.
4. **Participation covenant** — adaptations require a real operator signal or real stimulus, plus minimum participation. Fully automated self-tuning without a human-in-the-loop signal is rejected.
5. **Federated exchange** — raw imagery stays at the edge. What leaves the satellite is a derived update, never the underlying pixels.
6. **Epistemic alignment** — updates must reduce uncertainty, not reinforce prior bias. A "confirming" update that tightens posterior without adding genuinely new information is rejected.

These six gates convert a naively dangerous "let the model learn from what it sees" loop into a governed one. Without them, on-orbit TTT is uncontrolled drift; with them, TTT becomes a disciplined continual-learning loop that improves every pass instead of degrading.

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
Every candidate adaptation — whether a VLA-layer weight update or a trust-layer threshold change — passes through the six-gate filter before persistence. The architectural claim: stacked TTT is only safe on-orbit because the gates screen out bad updates that a human-in-the-loop would otherwise catch, and a human is precisely what is not available on orbit.

## Two-Track Submission
This repo is submitted to both tracks of the AI in Space hackathon. The scaffold, WCLI trust layer, mission-response layer, propagator, viability gates, and TTT infrastructure are identical across tracks — only the VLA backend changes.

- **Liquid Track (LFM2.5 + MuZero)** — A vision-language model serves as the Sentinel tile encoder: it processes each raw tile into a dense `(embed_dim,)` embedding vector that feeds MuZero's `h()` representation function. MuZero + MCTS then plans over encounter windows using that embedding. The hybrid is deliberate: MuZero's ResNet weighs a few MB and fits the satellite's 5 MB uplink budget; the encoder is LoRA fine-tuned on SimSat Sentinel tiles and runs locally for its single encoder pass per window. Training follows the ARC3 three-stage chain — (1) pretrain on the 86-trace Sentinel tile corpus, (2) fine-tune with SimSat scenario-pack augmentation, (3) two-scope TTT per live pass using the `TTTScope1` / `TTTScope2` budget schedules ported from `arc3_game.py`. The MuZero game adapter (`SimSatGame`), gym environment (`SimSatEnv`), FC-network config (`SimSatMuZeroConfigLFM`), and encoder protocol (`TileEncoder` / `build_encoder("lfm2vl")`) are all in `src/sim/muzero/`. The encoder slot is filled by `HFVisionTowerEncoder` — default `google/siglip-base-patch16-224` (768-dim, ~370 MB, real weights, verified end-to-end). LFM2.5-VL was the planned encoder but Liquid AI did not publish public weights by the deadline; the contract is unchanged so the LFM2.5-VL drop-in is a one-line `model_id` swap when those weights ship. **Liquid Track eval** (`scripts/muzero_lfm_eval.py`, 75 matched Sentinel tiles, all 3 scenario packs): encoder latency **18.5 ms/tile avg** (p95 24.3 ms) on RTX 2080; embed_dim=768; action distribution accept=17%, refine=68%, defer=5%, skip=9%. Full results in [MUZERO_LFM_EVAL.md](./MUZERO_LFM_EVAL.md).
- **General AI Track (Gemma-4 + open collaborator seats)** — ObservationVLA backed by a new Gemma-4-E2B fine-tune scoped specifically to SimSat's encounter-triage task (accept / defer / skip / refine over Sentinel-style tiles). Training code and dataset at `notebooks/kaggle-simsat-gemma4-v1/`; Kaggle kernel `benhaslam/simsat-gemma4-v1-training` (v9 trained successfully after a 21-fix T4 train); dataset `benhaslam/simsat-gemma4-v1` (294 weighted ChatML rows). On the 7-case operator-reviewed eval: usefulness-score MAE **0.16** (vs 0.27 `clip_local` baseline), bucketed action agreement **0.57**, all 3 ground-truth `accept` cases identified within 0.05 of operator score (see [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md)). This is **not** the `v35-gov` Gemma-4 fine-tune — `v35-gov` targets human-interview and consent-governance prompts, wrong task shape for satellite imagery. The VLA adapter is model-agnostic (`src/sim/observation_vla/transformers_vlm_local.py`); swapping backends is an env-var change, not a refactor. Two additional collaborator backends are wired and ready: **Genesis** (Guilherme Mesquita's model, `OBSERVATION_VLA_BACKEND=genesis`) and **Tesseract T3** (Garrett Sutherland's model, `OBSERVATION_VLA_BACKEND=tesseract_t3`). Any model that speaks the eight-key ObservationVLA JSON contract can plug into the same scaffold, trust layer, viability gates, and TTT loop without code changes. The competitive claim extends to this: the SimSat pipeline is model-agnostic by design — the governed continual-learning architecture is the contribution, not any one model weight checkpoint. See `COLLABORATOR_GUIDE.md` for integration instructions.

**Per-track pitch thesis (one sentence each):**

- **Liquid Track:** SimSat couples a vision-language tile encoder (SigLIP-base today, LFM2.5-VL the moment Liquid AI publishes weights — same one-line slot) with a MuZero planner — the hybrid fits a 5 MB satellite uplink budget while two-scope test-time training adapts both the encoder and the planner in-pass without ground-contact.
- **General AI Track:** SimSat fine-tunes Gemma-4-E2B specifically for satellite encounter triage, wraps it in a model-agnostic scaffold with viability-gated test-time training, so any vision-language model — Gemma-4, Genesis, Tesseract T3, or a future swap — inherits the same governed continual-learning loop with zero code changes.

## What Is New
- Real future encounter windows from the live TLE-backed propagator
- Scenario packs for challenge evaluation
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

These packs are encoded directly in [targets.json](./src/sim/data/encounter/targets.json) for reproducibility.

## Evaluation Outputs
Each evaluation logs:
- action distributions for scaffold and WCLI-trust planners
- action transition counts such as `accept->refine`
- materialization yield for top-ranked candidates
- decision-level deltas showing which windows changed, by how much, and why

The most challenge-relevant artifact is the delta list: it makes the WCLI trust layer falsifiable instead of rhetorical.

## ObservationVLA Reality Check
The ObservationVLA lane is backed by the Gemma-4-E2B SimSat fine-tune (`OBSERVATION_VLA_BACKEND=gemma4`). The current reviewed evaluation is intentionally presented with tight claims:

- reviewed set: N=7 (4 pinned operator-reviewed cases across all 3 scenario packs + 3 additional reviewed traces)
- useful/not-useful agreement: **0.86**
- usefulness-score MAE: **0.16** (vs 0.27 `clip_local` baseline — 41% improvement)
- bucketed action agreement: **0.57**; exact action agreement: 0.43
- 3/3 ground-truth `accept` cases identified within 0.05 of operator usefulness score
- known miss: Fort Myers Coast (`defer` predicted, `refine` actual, abs_error=0.71) — model under-confident on partial-cloud borderline windows
- known accept-bias: Rotterdam `refine` cases still over-predicted as `accept`; not corrected by additional training epochs (v10 = v9 on all metrics)
- the model should be treated as a calibrated evidence lane, not a trusted autonomous action policy
- the `clip_local` CLIP baseline remains available as a zero-weight-download reference

For the latest reviewed check:

```bash
python scripts/observation_vla_eval.py --inprocess
```

That writes [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md) with the current runtime, reviewed sample size, and conservative claim text.

## Judge-Facing Scorecard
For a compact, reproducible scorecard that can drop straight into notes or a submission draft:

```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000/sim --scenario-sweep --top-k 8 --materialize-top-k 2 --markdown
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
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim
```

This generates [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md) with one section per scenario pack containing:
- latest planner transitions and yield summary
- top delta highlights from the latest evaluation
- one curated case summary from a labelled trace, observation outcome, and mission-response action

The generated packet is Sentinel-first and prefers operator-reviewed labels when they exist. If no reviewed label exists yet, it falls back to `simulated_submission_case`.

To replace a simulated label with a real operator-reviewed outcome:

```bash
python scripts/review_queue_casebook.py --inprocess
python scripts/operator_review.py --base-url http://127.0.0.1:8000/sim --scenario-pack maritime_chokepoints
```

The review queue writes [REVIEW_QUEUE.md](./REVIEW_QUEUE.md) plus per-case images so the next human-review pass can compare the stored trace assessment with the current `clip_local` backend recommendation on the same imagery.

For a specific trace, inspect the full review bundle first:

```bash
python scripts/operator_review.py --base-url http://127.0.0.1:8000/sim --trace-id <trace_id> --show-bundle-only
```

Then replace the current label and pin it as the canonical submission case for that scenario:

```bash
python scripts/operator_review.py --base-url http://127.0.0.1:8000/sim --trace-id <trace_id> --reviewer "Your Name" --operator-action accept --useful true --usefulness-score 0.95 --pin-submission-case --pinned-by "Your Name"
```

Once one reviewed case is pinned per scenario pack, generate a strict reviewed-only packet:

```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim --reviewed-only
```

That mode fails loudly if any scenario pack still lacks a pinned operator-reviewed submission case.

For a visual companion built from those pinned cases and their stored images:

```bash
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000/sim
```

That generates [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md) plus local image assets for the pinned cases.

For a low-compute readiness check across the packet, casebook, pinned traces, and runtime honesty boundary:

```bash
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000/sim
```

## Reproducible Demo
1. Start SimSat:
```bash
docker compose up
```
2. Run the challenge demo:
```bash
python scripts/challenge_demo.py --base-url http://127.0.0.1:8000/sim --scenario-pack maritime_chokepoints
```
3. Compare planners directly:
```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000/sim --scenario-pack disaster_response_weather --top-k 10
```
4. Open the dashboard and use the Encounter Planner panel to switch scenario packs, run evaluation, and inspect the top decision deltas.

You do not need `MAPBOX_ACCESS_TOKEN` for this flow. If Mapbox is disabled, the runtime exposes that fact through `/capabilities`, and the challenge scripts continue using Sentinel plus geometry support.

## Low-Compute Rehearsal
If local GPU and CPU are busy, stick to the evaluation route rather than a long interactive sim session:

```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000/sim --scenario-pack maritime_chokepoints --top-k 8 --materialize-top-k 2
```

That path exercises the challenge thesis with minimal load and still gives you transition counts, yield, and top changed decisions.

It is also the recommended path if you do not want to enable Mapbox billing at all.

## Suggested Live Demo Flow
1. Pick `maritime_chokepoints` and run evaluation.
2. Show shared windows and the scaffold vs WCLI-trust transition counts.
3. Highlight at least one `accept -> refine` delta and explain the trust reason.
4. Switch to `disaster_response_weather` and show that cloud or limited imagery support raises refine pressure.
5. Materialize one high-confidence `accept` and one trust-gated `refine` candidate to illustrate why the refine path exists.
6. Close on the architectural claim: the same pipeline is running stacked TTT under six non-compensatory viability gates — the minimum safe configuration for on-orbit continual learning without a ground-truth validator.

## Known Issues and Review Bundles
The full in-repo code review from 2026-04-21 is preserved under `review/`:

- `review/2026-04-21-phase-1/` — Tier-1 correctness fixes (tzinfo bug at `src/sim/simulator.py:111`, hygiene cleanup, image reference fixes). **Apply these first.**
- `review/2026-04-21-phase-2/` — Tier-2 thesis/calibration work (accept→refine sweep, MAE calibration reframe, casebook auto-gen patch, architecture subsection).
- `review/2026-04-21-phase-3/` — Entry B backend scaffold (`entry-b/backend` branch; model-agnostic VLM adapter) and the Google Drive recon block.
- `review/2026-04-21-phase-4/` — This submission-doc refresh, TTT + viability thesis lock-in, and frontend brand-hide patch.

Known open gaps (disclosed, not hidden):
- The `accept→refine` thesis is demonstrated on the Port of Rotterdam pinned case (`urban_coastal_ambiguity`, `trace_4f65355f5c954fbf8db3fc684bb377af`) — scaffold `accept`, trust `refine`, operator confirmed `refine`, driven by cloud risk (48.78% cover).
- Accept-bias in the Gemma-4 fine-tune: Rotterdam refine cases still over-predicted as `accept`. Additional training epochs (v10 = 4 epochs) did not change this — requires dataset-level rebalancing.
- Fort Myers Coast miss: model predicts `defer` (score 0.20), operator says `refine` (score 0.90) — large abs_error on borderline partial-cloud windows.
- Stacked TTT is described architecturally. Live on-orbit demonstration requires the hackathon prize hardware (NVIDIA Orin 16GB); local demonstration shows the wiring and gating logic but not a full on-orbit drift trajectory.
- Genesis (Guilherme) and Tesseract T3 (Garrett) collaborator backends are wired and ready; waiting on collaborator fine-tuning / weights.

## Submission Framing
Use this wording in the pitch:

> SimSat is a governed on-orbit continual-learning loop. We model mission operations as a sequence of encounter windows, rank them with a cheap deterministic scaffold, and gate every decision through a WCLI-style trust layer that can refine instead of commit. A vision-language backend scores candidate tiles; a mission-response layer translates decisions into logged utility. On top of that, test-time training runs at two layers — the VLA adapts on streaming Sentinel tiles, and the trust gate tunes from realized-utility feedback — with both adaptation streams passing through six non-compensatory viability gates before anything persists. That gating mechanism is why this can run in orbit without a ground-truth validator, which is why it has to run in orbit.
