# SimSat Challenge Entry

## Thesis
SimSat is framed here as an encounter-planning system rather than only a continuous orbit simulator.

The entry compares two planners over the same future observation windows:
- `Scaffold`: deterministic geometry-and-availability scoring
- `WCLI Trust`: scaffold plus a trust-gated support layer that can `accept`, `defer`, `skip`, or `refine`

For the shortest reviewer-facing summary of the current state, use [SUBMISSION_BRIEF.md](/D:/SimSat/SUBMISSION_BRIEF.md).

The competitive claim is simple and testable:

> Treating observation opportunities as structured encounter episodes lets us reduce wasted high-cost materialization while preserving high-value windows through an explicit `refine` path.

This challenge entry is intentionally `no-Mapbox-safe`: the core planner, evaluation loop, and ObservationVLA trace pipeline do not require a Mapbox token.

## Two-Track Submission
This repo is submitted to both tracks of the AI in Space hackathon. The scaffold, WCLI trust layer, mission-response layer, propagator, and all submission scripts are identical across tracks — only the ObservationVLA model backend changes.

- **Liquid Track (Entry A)** — ObservationVLA backed by a Liquid Foundation Model (LFM2-VL or LFM2.5-VL, per the event's published eligibility list). Weights, training code, and a measurable improvement over the base model ship in-repo.
- **General AI Track (Entry B)** — ObservationVLA backed by a new Gemma-4 fine-tune scoped specifically to SimSat's encounter-triage task (accept / defer / skip / refine over Sentinel-style tiles). **This is not the v35-gov Gemma-4 fine-tune** that lives in our separate HAIC × Gemma-4 Good Kaggle project — v35-gov targets human-interview and consent-governance prompts, not satellite imagery, and would be the wrong task shape here.

Per-track model selection, fine-tune datasets, and the specific "why it must run in orbit" thesis statements are finalized in each track's entry document. The code path is the same; the ObservationVLA adapter is model-agnostic (see `src/sim/observation_vla/transformers_vlm_local.py`).

## What Is New
- Real future encounter windows from the live TLE-backed propagator
- Scenario packs for challenge evaluation
- Decision-level deltas between scaffold and WCLI-trust planners
- Materialization-yield comparison across the same ranked windows
- ObservationVLA lane with image-conditioned assessment, labeled trace capture, and lightweight pass/scenario/mission calibration hooks
- Explicit runtime capability reporting so local runs can verify that Mapbox is optional
- Mission-response layer that turns effective observation decisions into concrete downstream actions and simulated utility

The current ObservationVLA backend evaluation is tracked separately in [OBSERVATION_VLA_EVAL.md](/D:/SimSat/OBSERVATION_VLA_EVAL.md).

Core implementation lives in:
- [service.py](/D:/SimSat/src/sim/encounter/service.py)
- [planner.py](/D:/SimSat/src/sim/encounter/planner.py)
- [trust_model.py](/D:/SimSat/src/sim/encounter/trust_model.py)
- [targets.json](/D:/SimSat/src/sim/data/encounter/targets.json)

## Scenario Packs
- `maritime_chokepoints`: Suez, Panama, Singapore
- `disaster_response_weather`: flood/storm-sensitive coastal and industrial targets
- `urban_coastal_ambiguity`: dense mixed-use port and shoreline scenes where tempting windows may still deserve refinement

These packs are encoded directly in [targets.json](/D:/SimSat/src/sim/data/encounter/targets.json) for reproducibility.

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

For the latest reviewed check:

```bash
python scripts/observation_vla_eval.py --inprocess
```

That writes [OBSERVATION_VLA_EVAL.md](/D:/SimSat/OBSERVATION_VLA_EVAL.md) with the current runtime, reviewed sample size, and conservative claim text.

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

This generates [SUBMISSION_PACKET.md](/D:/SimSat/SUBMISSION_PACKET.md) with one section per scenario pack containing:
- latest planner transitions and yield summary
- top delta highlights from the latest evaluation
- one curated case summary from a labelled trace, observation outcome, and mission-response action

The generated packet is Sentinel-first and prefers operator-reviewed labels when they exist. If no reviewed label exists yet, it falls back to `simulated_submission_case`.

To replace a simulated label with a real operator-reviewed outcome:

```bash
python scripts/review_queue_casebook.py --inprocess
python scripts/operator_review.py --base-url http://127.0.0.1:8000/sim --scenario-pack maritime_chokepoints
```

The review queue writes [REVIEW_QUEUE.md](/D:/SimSat/REVIEW_QUEUE.md) plus per-case images so the next human-review pass can compare the stored trace assessment with the current `clip_local` backend recommendation on the same imagery.

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

That generates [SUBMISSION_CASEBOOK.md](/D:/SimSat/SUBMISSION_CASEBOOK.md) plus local image assets for the pinned cases.

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

## Known Issues and Review Bundles
The full in-repo code review from 2026-04-21 is preserved under `review/`:

- `review/2026-04-21-phase-1/` — Tier-1 correctness fixes (tzinfo bug at `src/sim/simulator.py:111`, hygiene cleanup, image reference fixes). **Apply these first.**
- `review/2026-04-21-phase-2/` — Tier-2 thesis/calibration work (accept→refine sweep, MAE calibration reframe, casebook auto-gen patch, architecture subsection).
- `review/2026-04-21-phase-3/` — Entry B backend scaffold (`entry-b/backend` branch; model-agnostic VLM adapter) and the Google Drive recon block.
- `review/2026-04-21-phase-4/` — This submission-doc refresh + frontend HAIC-hide patch.

Known open gaps at submission time (disclosed, not hidden):
- All 3 pinned casebook cases are `accept→accept`. The `accept→refine` thesis is logged in eval outputs but is not yet demonstrated on a pinned demo case.
- Sweep generation (`REVIEW_SET_BUILD.md`) reports `quality_skips=60, added_distinct=0` across all 3 targets — either a dedup bug or genuinely deterministic SSO windows. Under investigation.
- Pinned casebook traces show `Observation runtime: stub`; current backend is `clip_local`. Re-materialization under `clip_local` is in-progress (see `review/2026-04-21-phase-2/S3`).

## Submission Framing
Use this wording in the pitch:

> We model mission operations as a sequence of encounter windows rather than only continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. ObservationVLA adds image-conditioned reassessment, and a mission-response layer converts those judgments into concrete downstream actions with explicit utility. This produces a transparent, auditable mission-planning loop that improves materialization yield and exposes uncertainty instead of hiding it.
