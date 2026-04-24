# SimSat Submission Brief

## Claim
SimSat is a governed on-orbit continual-learning loop. It treats mission operations as a sequence of encounter windows rather than continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. A Sentinel-first ObservationVLA lane performs image-conditioned reassessment, and a mission-response layer converts those judgments into explicit downstream actions with logged utility. **Test-time training runs at two layers simultaneously** — the VLA backend adapts on streaming Sentinel tiles, and the trust layer tunes online from realized-utility feedback — with **both adaptation streams gated by six non-compensatory viability checks** before any update is allowed to persist.

## Why this must run in orbit
Three stacked constraints force the full pipeline to the spacecraft, in priority order:

1. **Distribution shift without a ground-truth validator.** A satellite streaming imagery hits real drift — new geographies, seasonal variation, cloud and sensor conditions — and cannot call home for a label. SimSat's answer is six non-compensatory viability gates that every candidate observation must pass before updating system state: entropy reduction (must decrease uncertainty), extraction risk (no bulk scraping), PRISM consistency (claimed metadata matches measured properties), participation covenant (real stimulus, minimum operator participation), federated exchange (raw imagery stays at edge), and epistemic alignment (updates reduce uncertainty rather than reinforce bias). Without these gates, on-orbit TTT is uncontrolled drift; with them, it is a governed continual-learning loop. This is the mechanism the rest of the competition will not have.
2. **Bandwidth.** A 5 MB uplink and 10 MB downlink cannot ship model weight updates to ground and back at any useful cadence. TTT that tracks drift must happen on the satellite itself.
3. **Latency.** The next encounter window arrives in minutes. Adaptations must be usable in the very next pass; there is no round-trip budget for ground re-training and re-upload.

## Tracks
SimSat is submitted to both tracks of the AI in Space hackathon. Both tracks run identical scaffold, WCLI trust, mission-response, viability-gate, and stacked-TTT infrastructure. What differs is the VLA backend:

- **Liquid Track (LFM2.5)** — a Liquid Foundation Model serves as the image-conditioned evidence scorer and as the target of VLA-layer test-time training. Weights, training code, and a measurable improvement over the base model ship in-repo.
- **General AI Track (Gemma-4)** — a Gemma-4 fine-tune scoped specifically to SimSat's encounter-triage task (accept / defer / skip / refine over Sentinel-style tiles). This is **not** the `v35-gov` Gemma-4 fine-tune that lives in our separate HAIC × Gemma-4 Good Kaggle project — `v35-gov` targets human-interview / consent-governance prompts, wrong task shape for satellite imagery. The Entry B VLA adapter is model-agnostic (`src/sim/observation_vla/transformers_vlm_local.py`); swapping the final choice is an env-var change, not a refactor.

Per-track thesis statements for the pitch are finalized in [CHALLENGE_ENTRY.md](./CHALLENGE_ENTRY.md).

## Evidence
- Reviewed submission packet: [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md)
- Visual casebook: [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md)
- Readiness checklist: [SUBMISSION_READINESS.md](./SUBMISSION_READINESS.md)
- ObservationVLA reviewed eval: [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md)

Current reviewed cases:
- `maritime_chokepoints` → Suez Canal, reviewer `Ben Haslam`, usefulness `0.95`
- `disaster_response_weather` → Houston Ship Channel, reviewer `Ben Haslam`, usefulness `0.92`
- `urban_coastal_ambiguity` → Port of Rotterdam, reviewer `Ben Haslam`, usefulness `0.90`

## Runtime Truth
- Sentinel is the primary observation source.
- Mapbox is optional and disabled in the current submission flow.
- ObservationVLA runs in `clip_local` mode with `openai/clip-vit-base-patch32` as the baseline backend during local testing; the track-specific backends (LFM2.5, Gemma-4) plug into the same adapter interface.
- **TTT is active at two layers:** VLA weights / adapters adapt on streaming tiles; the WCLI trust layer thresholds and priors update from realized-utility feedback. Both adaptation streams flow through the six viability gates before persistence.
- The current reviewed eval is still low-N. Useful/not-useful alignment is 1.00 on the 3 reviewed cases, but magnitude calibration has MAE 0.27 — the model is systematically under-confident vs. the operator. Downstream actions consume the binary agreement, not the raw score. See `OBSERVATION_VLA_EVAL.md` for the numbers.
- Mission-response is a policy-and-utility layer, not live spacecraft actuation.

## Demo Order
1. Open [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md) for the scorecard and reviewed cases.
2. Open [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md) for the three pinned visual examples.
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
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim --reviewed-only
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000/sim
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000/sim
```
