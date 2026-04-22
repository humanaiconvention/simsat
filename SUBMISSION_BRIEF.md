# SimSat Submission Brief

## Claim
SimSat treats mission operations as a sequence of encounter windows rather than only continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. A Sentinel-first ObservationVLA lane performs image-conditioned reassessment, and a mission-response layer converts those judgments into explicit downstream actions with logged utility.

## Evidence
- Reviewed submission packet: [D:\SimSat\SUBMISSION_PACKET.md](D:\SimSat\SUBMISSION_PACKET.md)
- Visual casebook: [D:\SimSat\SUBMISSION_CASEBOOK.md](D:\SimSat\SUBMISSION_CASEBOOK.md)
- Readiness checklist: [D:\SimSat\SUBMISSION_READINESS.md](D:\SimSat\SUBMISSION_READINESS.md)
- ObservationVLA reviewed eval: [D:\SimSat\OBSERVATION_VLA_EVAL.md](D:\SimSat\OBSERVATION_VLA_EVAL.md)

Current reviewed cases:
- `maritime_chokepoints` -> Suez Canal, reviewer `Ben Haslam`, usefulness `0.95`
- `disaster_response_weather` -> Houston Ship Channel, reviewer `Ben Haslam`, usefulness `0.92`
- `urban_coastal_ambiguity` -> San Francisco Bay, reviewer `Ben Haslam`, usefulness `0.90`

## Runtime Truth
- Sentinel is the primary observation source.
- Mapbox is optional and disabled in the current submission flow.
- ObservationVLA now runs in `clip_local` mode with `openai/clip-vit-base-patch32`.
- The current reviewed eval is still low-N and should be read narrowly: usefulness alignment is promising, exact operator-action matching is not.
- Mission-response is a policy-and-utility layer, not live spacecraft actuation.

## Demo Order
1. Open [D:\SimSat\SUBMISSION_PACKET.md](D:\SimSat\SUBMISSION_PACKET.md) for the scorecard and reviewed cases.
2. Open [D:\SimSat\SUBMISSION_CASEBOOK.md](D:\SimSat\SUBMISSION_CASEBOOK.md) for the three pinned visual examples.
3. Use [D:\SimSat\CHALLENGE_ENTRY.md](D:\SimSat\CHALLENGE_ENTRY.md) for the spoken walkthrough and architecture framing.

## Reproduce
```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim --reviewed-only
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000/sim
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000/sim
```
