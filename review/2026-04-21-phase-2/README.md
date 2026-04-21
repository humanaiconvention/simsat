# SimSat Review — Phase 2 (Should-Do)

**Date:** 2026-04-21
**Follows:** [`review/2026-04-21-phase-1/`](../2026-04-21-phase-1/)
**Scope:** 4 Should-Do items that tighten the thesis narrative, the AI story, and the demo polish — but do not block submission if skipped.

| # | Folder | What it fixes | Effort |
|---|---|---|---|
| S1 | [`S1-accept-refine-delta/`](./S1-accept-refine-delta/) | All 3 pinned submission cases are `accept → accept`, so the WCLI-trust thesis is rhetorical, not falsified. Ship a sweep script + narrative addendum that surfaces at least one `accept → refine` delta for the live demo. | 1–2 hours |
| S2 | [`S2-mae-calibration/`](./S2-mae-calibration/) | `OBSERVATION_VLA_EVAL.md` reports `MAE 0.27` without framing. Reframe it as a calibration point with explicit operational accommodation. | 10 min |
| S3 | [`S3-rematerialize-clip-local/`](./S3-rematerialize-clip-local/) | `SUBMISSION_CASEBOOK.md` shows `Observation runtime: stub` for all 3 pinned cases even though the backend is now `clip_local`. Re-assess and re-pin so the casebook reflects current runtime. | 30 min |
| S4 | [`S4-architecture-explorer/`](./S4-architecture-explorer/) | Phase 1 adds the Architecture Explorer to the pitch. Phase 2 adds a dedicated subsection in `CHALLENGE_ENTRY.md` that names every model visualized and what the comparison shows. | 15 min |

## Apply order

Any order works, but recommended:

1. **S2 first** (10 min doc edit) — ships a cleaner reviewed-eval story that other items can refer to.
2. **S4** (15 min doc edit) — lands the Architecture Explorer detail while the framing from Phase 1 is fresh.
3. **S3** (30 min, needs Docker) — requires the sim stack running to re-assess traces.
4. **S1** (1–2 hours, discovery-dependent) — runs a sweep; if no `accept → refine` surfaces naturally, there's a fallback using a temporary policy tweak.

## What's different from Phase 1

Phase 1 items were either pure code fixes (M1, M2) or prose rewrites with no external dependencies (M3, M4). Phase 2 items S1 and S3 **require a running local Docker stack** (`docker compose up`) to execute because they touch the live simulator/encounter service. The artifacts in this folder are scripts, commands, and prose — Ben runs them locally.
