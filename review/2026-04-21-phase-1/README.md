# SimSat Review — Phase 1 (Must-Do)

**Date:** 2026-04-21
**Scope:** 4 Must-Do items for the DPhi Space × Liquid AI "AI in Space" Hackathon submission
**Source review:** full repository audit — propagator, encounter pipeline, HAIC convention layer, Django backend, React frontend, all submission docs

This folder contains copy-paste-ready artifacts for the 4 Must-Do items. Each subfolder has its own README explaining the change, the rationale, and how to apply it to `D:\SimSat`.

## Apply order

| # | Folder | What it fixes | Effort |
|---|---|---|---|
| M1 | [`M1-timezone-fix/`](./M1-timezone-fix/) | `simulator.py:111` emits local-time ISO strings → phase-shifted encounter windows in any non-UTC environment | 10 min |
| M2 | [`M2-cleanup/`](./M2-cleanup/) | 6 stray `tmp*_simsat_api.py` files + broken standalone `gui.py` in the source tree | 5 min |
| M3 | [`M3-haic-narrative/`](./M3-haic-narrative/) | `SUBMISSION_BRIEF.md` and `CHALLENGE_ENTRY.md` don't mention HAIC, yet the dashboard foregrounds it and the Architecture Explorer visualizes it alongside Liquid AI's LFM2-8b | 15 min |
| M4 | [`M4-review-notes/`](./M4-review-notes/) | The 3 pinned review notes all say "user wouldn't know X but appears correct" — undermines reviewer credibility. Rewritten with observable features. | 20 min |

## After Phase 1

Once these are applied and pushed, Phase 2 (Should-Do) covers:

- **S1** — Generate one `accept → refine` demo case so the WCLI-trust thesis is falsified on-demo, not just in theory
- **S2** — Frame the ObservationVLA MAE=0.27 as a calibration point, not a weakness
- **S3** — Re-materialize the 3 pinned cases under `clip_local` so the casebook matches current backend
- **S4** — Extend `CHALLENGE_ENTRY.md` with an Architecture Explorer subsection naming each model visualized

## What I could not do from this environment

I'm in a Linux sandbox without access to your `D:\SimSat` filesystem. All artifacts here are copy-paste-ready but must be applied locally. Phase 2 item S3 (re-materialize pinned cases) requires your local Docker stack to be running — I ship those as commands-to-run.
