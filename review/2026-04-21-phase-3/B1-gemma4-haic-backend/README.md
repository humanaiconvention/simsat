# Phase 3 B1 — Entry B (Gemma-4 + HAIC v35-gov) backend scaffold

**Status:** shipped to feature branch `entry-b/backend` (not merged to main)
**Purpose:** wire HAIC v35-gov fine-tune into the SimSat sim service as a second ObservationVLA backend alongside the existing `clip_local`.

## What's on the feature branch

```
entry-b/backend
├── src/sim/observation_vla/
│   ├── gemma4_haic_local.py      # NEW: Gemma4HAICAdapter class
│   └── README_ENTRY_B.md         # NEW: integration guide + payload contract
└── scripts/
    ├── download_haic_v35_gov.sh       # Kaggle CLI downloader (POSIX)
    ├── download_haic_v35_gov.ps1      # Kaggle CLI downloader (Windows)
    └── download_haic_v35_gov.README.md  # Weight-download instructions
```

## Why a feature branch (not main)

Ben said "not ready to submit yet, need to test it." The feature branch keeps `main` safe for the current Phase 1/2 review artifacts while Entry B is being verified.

Merge to `main` only after:
1. `scripts/download_haic_v35_gov.sh` successfully populates `./weights/haic-v35-gov/` from Kaggle.
2. `docker compose up` with `OBSERVATION_VLA_BACKEND=gemma4_haic_local` starts cleanly.
3. `python scripts/haic_test.py` passes all 11 sections against the new backend.
4. `python scripts/observation_vla_eval.py --inprocess` produces a new `OBSERVATION_VLA_EVAL.md` with acceptable agreement on the 3 pinned cases.
5. LoRA adapter size confirmed ≤ 5 MB for the "satellite upload budget" pitch.

## Shareable checkpoint for Guilherme

Once the merge gate above is green:
- Add `guilhermeferraribr` as collaborator on `benhaslam/haic-gemma4-v35-gov-unsloth` (Kaggle kernel settings → Sharing)
- Same for `haic-gemma4-v35-gov-quantize` and `haic-gemma4-v35-gov-dataset-generator`
- Push `entry-b/backend` branch → open PR against `main` with the README_ENTRY_B.md as the description

## Parallel: Entry A scaffold

Entry A (LFM2-VL for the Liquid Track) will need the same shape:
- `src/sim/observation_vla/lfm2_vl_local.py` (new file)
- `scripts/download_lfm2_vl.sh` (downloader — either HuggingFace or Kaggle if Ben trains one there)

The Drive search to find existing LFM2-VL Colab notebooks was **blocked** (see `review/2026-04-21-phase-3/B2-drive-recon/README.md`) — needs Google Drive scope re-auth before I can look.
