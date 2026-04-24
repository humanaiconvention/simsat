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

- **Liquid Track (LFM2.5)** — ObservationVLA backed by a Liquid Foundation Model (LFM2.5, per the event's published eligibility list). Weights, training code, and a measurable improvement over the base model ship in-repo, plus documentation of the live on-orbit TTT adaptation on streaming Sentinel tiles.
- **General AI Track (Gemma-4)** — ObservationVLA backed by a new Gemma-4 fine-tune scoped specifically to SimSat's encounter-triage task (accept / defer / skip / refine over Sentinel-style tiles). This is **not** the `v35-gov` Gemma-4 fine-tune that lives in our separate HAIC × Gemma-4 Good Kaggle project — `v35-gov` targets human-interview and consent-governance prompts, wrong task shape for satellite imagery. The VLA adapter is model-agnostic (`src/sim/observation_vla/transformers_vlm_local.py`); swapping the final model choice is an env-var change, not a refactor.

Per-track one-sentence thesis statements for the pitch video are finalized as each entry ships.

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
The ObservationVLA lane is now image-model-backed via a local `clip_local` backend rather than a pure stub. That is real progress, but the current reviewed evaluation is intentionally presented with tight claims:

- the reviewed set is still very small (N=3 pinned cases)
- useful/not-useful alignment is strong on the reviewed set (1.00 action and usefulness agreement)
- magnitude calibration is weak: usefulness-score MAE is 0.27, with the model systematically under-confident vs. the human operator (model ~0.66, operator 0.90–0.95)
- downstream actions consume the binary agreement, not the raw magnitude — so the MAE does not change planner behaviour, but it is a rigor gap we are not hiding
- the visual model should currently be treated as an image-conditioned evidence lane, not a trusted autonomous action policy
- the `clip_local` baseline is a reference backend; the track-specific LFM2.5 and Gemma-4 fine-tunes replace it in their respective entries and will show a measurable improvement over the baseline per the Liquid Track eligibility rules

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

Known open gaps at submission time (disclosed, not hidden):
- All 3 pinned casebook cases are `accept→accept`. The `accept→refine` thesis is logged in eval outputs but is not yet demonstrated on a pinned demo case.
- Sweep generation (`REVIEW_SET_BUILD.md`) reports `quality_skips=60, added_distinct=0` across all 3 targets — either a dedup bug or genuinely deterministic SSO windows. Under investigation.
- Pinned casebook traces show `Observation runtime: stub`; current backend is `clip_local`. Re-materialization under `clip_local` is in-progress (see `review/2026-04-21-phase-2/S3`).
- Stacked TTT is described architecturally in this entry. Live on-orbit demonstration requires the hackathon prize hardware (NVIDIA Orin 16GB in-space compute); local demonstration shows the wiring and the gating logic but not a full on-orbit drift trajectory.

## Submission Framing
Use this wording in the pitch:

> SimSat is a governed on-orbit continual-learning loop. We model mission operations as a sequence of encounter windows, rank them with a cheap deterministic scaffold, and gate every decision through a WCLI-style trust layer that can refine instead of commit. A vision-language backend scores candidate tiles; a mission-response layer translates decisions into logged utility. On top of that, test-time training runs at two layers — the VLA adapts on streaming Sentinel tiles, and the trust gate tunes from realized-utility feedback — with both adaptation streams passing through six non-compensatory viability gates before anything persists. That gating mechanism is why this can run in orbit without a ground-truth validator, which is why it has to run in orbit.
