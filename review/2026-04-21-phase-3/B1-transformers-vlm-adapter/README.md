# Phase 3 B1 (revised) — Model-agnostic TransformersVLMAdapter

**Status:** shipped to feature branch `entry-b/backend` (not merged to main)
**Supersedes:** `B1-gemma4-haic-backend/` (deleted in same commit — the original `Gemma4HAICAdapter` was hard-wired to HAIC v35-gov, which turned out to be a human-interview model unsuitable for SimSat).

## Why the refactor

Investigation of the HAIC v35-gov Kaggle kernel (`benhaslam/haic-gemma4-v35-gov-unsloth`) and its outputs (`haic_v35_gov_full_results.json`) revealed:

- **v35-gov is a convention-grounded human-interview model** — its SGT scenarios pivot on emotional user input (`[PIVOT: COUNTERFACTUAL]`, `[PIVOT: TEMPORAL]`).
- **It has no training signal for satellite imagery triage** — the exact task SimSat's ObservationVLA needs.
- **v35-gov belongs to a separate submission** — the Kaggle "HAIC × Gemma-4 Good" hackathon, not DPhi Space × Liquid AI.

The original `Gemma4HAICAdapter` class was therefore retired. `TransformersVLMAdapter` retains the same scaffold structure (env-var-driven model loading, 3 modes: lora/merged/gguf, same payload shape) but is model-agnostic — it accepts any transformers-compatible model via generic `OBSERVATION_VLM_*` env vars.

## What's on the feature branch now

```
entry-b/backend
└── src/sim/observation_vla/
    ├── transformers_vlm_local.py   # NEW: TransformersVLMAdapter, model-agnostic
    └── README_ENTRY_B.md           # UPDATED: env var contract, config examples, track plan
```

Removed:
- `src/sim/observation_vla/gemma4_haic_local.py`
- `scripts/download_haic_v35_gov.sh`, `.ps1`, `.README.md` (v35-gov-specific)
- `review/2026-04-21-phase-3/B1-gemma4-haic-backend/README.md` (this file supersedes it)

## Model assignments per track (as of 2026-04-21)

| Track | Entry | Model | Notes |
|---|---|---|---|
| Liquid Track | A | **LFM2.5-VL** preferred; LFM2-VL fallback | Lu.ma page explicitly lists both as eligible; judging doc only names LFM2-VL in the fine-tuning clause (inconsistency — Lu.ma is authoritative on eligibility) |
| General AI Track | B | **Gemma-4 fine-tune** specifically trained for SimSat | New dataset + training run required; not to be confused with v35-gov (separate hackathon) |

Both models will load through the same adapter by swapping env vars. No further code changes needed on the adapter side.

## Pending decisions

1. **Ben × Guilherme meeting** — lock in the one-sentence thesis for Entry B's pitch.
2. **Dataset generation for Gemma-4-SimSat training** — mirror the approach used for v35-gov's dataset generator kernel, but with SimSat-specific triage tasks (Sentinel image + target metadata → triage action).
3. **LFM2.5-VL vs LFM2-VL final call** — confirm via Liquid AI Discord (#ai-in-space-hackathon) that LFM2.5-VL is acceptable, then commit.

## Pre-merge gate

Unchanged from the prior revision — see `src/sim/observation_vla/README_ENTRY_B.md`.
