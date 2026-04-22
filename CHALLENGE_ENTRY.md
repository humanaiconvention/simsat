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
- the reviewed set is still very small
- useful/not-useful alignment is stronger than direct action agreement
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

## Submission Framing
Use this wording in the pitch:

> We model mission operations as a sequence of encounter windows rather than only continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. ObservationVLA adds image-conditioned reassessment, and a mission-response layer converts those judgments into concrete downstream actions with explicit utility. This produces a transparent, auditable mission-planning loop that improves materialization yield and exposes uncertainty instead of hiding it.
