# SimSat Submission Brief

## Claim
SimSat treats mission operations as a sequence of encounter windows rather than only continuous propagation. A deterministic scaffold ranks windows cheaply, then a WCLI-style trust layer decides whether to accept, defer, skip, or refine before expensive imagery materialization. A Sentinel-first ObservationVLA lane performs image-conditioned reassessment, and a mission-response layer converts those judgments into explicit downstream actions with logged utility.

## Tracks
This repo is submitted to both tracks of the AI in Space hackathon:

- **Liquid Track** — uses the Liquid Foundation Model (LFM2-VL / LFM2.5-VL) as the ObservationVLA backend. Weights, training code, and fine-tune methodology are public in-repo when Entry A ships.
- **General AI Track** — uses a new Gemma-4 fine-tune scoped specifically to SimSat triage. (Note: this is **not** the v35-gov fine-tune that lives in the separate HAIC × Gemma-4 Good project — v35-gov is a human-interview / consent-governance model, wrong task shape for satellite imagery.)

The shared scaffold, WCLI trust layer, mission-response layer, and submission infrastructure are the same across both tracks. Entry-specific model details live in [CHALLENGE_ENTRY.md](/D:/SimSat/CHALLENGE_ENTRY.md).

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
- The current reviewed eval is still low-N and should be read narrowly: useful/not-useful alignment is 1.00 on the 3 reviewed cases, but magnitude calibration (usefulness score) has MAE 0.27 — the model is systematically under-confident vs. the human operator. Downstream actions use the binary agreement, not the raw score. See `OBSERVATION_VLA_EVAL.md` for the numbers.
- Mission-response is a policy-and-utility layer, not live spacecraft actuation.

## Demo Order
1. Open [D:\SimSat\SUBMISSION_PACKET.md](D:\SimSat\SUBMISSION_PACKET.md) for the scorecard and reviewed cases.
2. Open [D:\SimSat\SUBMISSION_CASEBOOK.md](D:\SimSat\SUBMISSION_CASEBOOK.md) for the three pinned visual examples.
3. Use [D:\SimSat\CHALLENGE_ENTRY.md](D:\SimSat\CHALLENGE_ENTRY.md) for the spoken walkthrough, architecture framing, and per-track model notes.

## Known Issues
Code-review findings and their patch bundles live in-repo under `review/`:

- `review/2026-04-21-phase-1/` — Must-Do correctness fixes (tzinfo bug, hygiene)
- `review/2026-04-21-phase-2/` — Should-Do thesis + calibration work
- `review/2026-04-21-phase-3/` — Entry B scaffold + Drive recon block
- `review/2026-04-21-phase-4/` — Submission-docs refresh (this refresh)
- `KNOWN_ISSUES.md` (if present at repo root) — consolidated index across all phases

## Reproduce
```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim --reviewed-only
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000/sim
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000/sim
```
