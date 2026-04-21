# SimSat Submission Brief

## Claim
SimSat treats mission operations as a sequence of encounter windows rather than only continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. A Sentinel-first ObservationVLA lane performs image-conditioned reassessment, and a mission-response layer converts those judgments into explicit downstream actions with logged utility.

## Convention Layer

Above the encounter planner sits the **HAIC (Human AI Convention) layer**. Where the encounter pipeline produces decisions, the HAIC layer produces *provable* decisions: every stimulus bundle, every interview turn, every viability gate, and every PRISM entropy measurement is sealed into a Merkle-rooted receipt (seven leaf commitments, SHA-256 root). The same Earth-imagery stream feeds both — the encounter pipeline treats the imagery as mission input, HAIC treats it as human-inspectable evidence, and both emit auditable traces.

This matters for the AI-in-Space setting because the hard problem is not computing a window. It is trusting the model that decides whether to use it. SimSat ships an **Architecture Explorer** that visualizes HAIC v3–v8 alongside **Liquid AI's LFM2-8b and LFM2-8b-a1b**, Gemma 3/4, Llama 3.1, and Ministral, so the model landscape the trust layer operates over is legible rather than a black box. The convention is deliberately provider-neutral: today's interviewer runs against Anthropic via `ANTHROPIC_API_KEY`, tomorrow's could run against LFM2 on-orbit.

## Evidence
- Reviewed submission packet: [D:\SimSat\SUBMISSION_PACKET.md](D:\SimSat\SUBMISSION_PACKET.md)
- Visual casebook: [D:\SimSat\SUBMISSION_CASEBOOK.md](D:\SimSat\SUBMISSION_CASEBOOK.md)
- Readiness checklist: [D:\SimSat\SUBMISSION_READINESS.md](D:\SimSat\SUBMISSION_READINESS.md)
- ObservationVLA reviewed eval: [D:\SimSat\OBSERVATION_VLA_EVAL.md](D:\SimSat\OBSERVATION_VLA_EVAL.md)
- Architecture Explorer: open the dashboard at http://localhost:8000 and click **Architecture Explorer** in the header.

Current reviewed cases:
- `maritime_chokepoints` -> Suez Canal, reviewer `Ben Haslam`, usefulness `0.95`
- `disaster_response_weather` -> Houston Ship Channel, reviewer `Ben Haslam`, usefulness `0.92`
- `urban_coastal_ambiguity` -> San Francisco Bay, reviewer `Ben Haslam`, usefulness `0.90`

## Runtime Truth
- Sentinel is the primary observation source.
- Mapbox is optional and disabled in the current submission flow.
- ObservationVLA now runs in `clip_local` mode with `openai/clip-vit-base-patch32`.
- The current reviewed eval is still low-N and should be read narrowly: usefulness alignment is promising, exact operator-action matching is not yet broadly validated.
- Mission-response is a policy-and-utility layer, not live spacecraft actuation.
- HAIC receipts are cryptographic artifacts of decisions. They are not themselves spacecraft actuation; they are how an operator proves *why* a decision was made.

## Demo Order
1. Open [D:\SimSat\SUBMISSION_PACKET.md](D:\SimSat\SUBMISSION_PACKET.md) for the scorecard and reviewed cases.
2. Open [D:\SimSat\SUBMISSION_CASEBOOK.md](D:\SimSat\SUBMISSION_CASEBOOK.md) for the three pinned visual examples.
3. Open the dashboard: run the Encounter Planner, then open **Architecture Explorer** and walk through the model landscape (HAIC v3–v8 → LFM2 → Gemma 4 → Llama 3.1 → Ministral).
4. Use [D:\SimSat\CHALLENGE_ENTRY.md](D:\SimSat\CHALLENGE_ENTRY.md) for the spoken walkthrough and architecture framing.

## Reproduce
```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim --reviewed-only
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000/sim
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000/sim
```
