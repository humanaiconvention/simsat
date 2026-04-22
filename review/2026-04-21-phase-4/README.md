# review/2026-04-21-phase-4 — Submission docs refresh + TTT thesis + viability-gate mechanism

Generated 2026-04-21. Amended later the same day to lock in the TTT thesis, finalize the two-track model selection, and resolve the HAIC narrative into a brand-hide + mechanism-feature stance.

## What this bundle is

Hand-edited updates to the submission-facing documents after the full-repo review completed earlier today, plus a frontend patch that keeps the HAIC brand out of the dashboard header. The bundle does **not** include auto-generated files (`SUBMISSION_PACKET.md`, `SUBMISSION_READINESS.md`, `REVIEW_QUEUE.md`, `REVIEW_SET_BUILD.md`) — those regenerate from scripts after the underlying fixes land locally. See `S6-autogen-regen/` for the regeneration checklist.

## Decisions baked into this bundle

1. **HAIC brand out of pitch and UI. Viability mechanism featured in pitch.** The six-gate non-compensatory validation convention (entropy reduction, extraction risk, PRISM consistency, participation covenant, federated exchange, epistemic alignment) is our technical differentiator and leads the "why must run in orbit" argument. We describe it **mechanically** in submission prose; we do **not** name "HAIC" in pitch surfaces. The dashboard badge and Architecture Explorer link stay hidden per `S5-frontend-haic-hide/`. Repo source (`src/sim/haic/viability.py`) keeps its internal naming — judges who clone and grep will find it.

2. **Stacked TTT is the central technical thesis.** Test-time training runs at two layers simultaneously:
   - **VLA-layer TTT** — the vision-language backend (LFM2.5 in Entry A, Gemma-4 in Entry B) adapts on streaming Sentinel tiles via lightweight adapters / weight updates.
   - **Trust-layer TTT** — the WCLI gate tunes thresholds and priors online from realized-utility feedback post-materialization.
   Both adaptation streams pass through the six viability gates before anything persists. Architectural claim: this is the minimum safe configuration for on-orbit continual learning without a ground-truth validator.

3. **Two tracks by VLA backend:**
   - **Liquid Track (LFM2.5)** — canonical label. Drops the earlier "LFM2-VL fallback" language.
   - **General AI Track (Gemma-4)** — new SimSat-specific fine-tune. **Not** the `v35-gov` model that lives in the separate HAIC × Gemma-4 Good Kaggle project — `v35-gov` is a human-interview / consent-governance fine-tune, wrong task shape for satellite imagery.
   Both tracks run identical scaffold, WCLI trust, mission-response, viability gates, and TTT infrastructure.

4. **"Why must run in orbit" argument, priority order:**
   1. Distribution shift without a ground-truth validator — six viability gates solve on-orbit continual learning. This is the unique angle.
   2. Bandwidth — 5 MB up / 10 MB down cannot ferry model weight updates to ground and back at any useful cadence.
   3. Latency — adaptation must land in-pass before the next encounter window arrives.

## Contents

| Sub-bundle | What it touches | Kind |
|---|---|---|
| `S1-submission-brief/` | `SUBMISSION_BRIEF.md` | Doc rewrite (amended with TTT + viability + track labels) |
| `S2-challenge-entry/` | `CHALLENGE_ENTRY.md` | Doc rewrite (amended with TTT + viability + track labels) |
| `S3-casebook-notes/` | `SUBMISSION_CASEBOOK.md` + `operator_review.py` commands | Doc rewrite + script invocations |
| `S4-root-readme/` | top-level `README.md` | Patch description |
| `S5-frontend-haic-hide/` | `dashboard/frontend/src/App.tsx` | Frontend patch (unchanged by the mechanism-only pivot) |
| `S6-autogen-regen/` | `submission_evidence.py`, `submission_readiness.py`, `review_queue_casebook.py` | Regeneration checklist |
| `S7-known-issues-consolidated/` | `KNOWN_ISSUES.md` (new file) | New consolidation (amended to reflect the mechanism-only HAIC stance + TTT demonstration-scope disclosure) |

## Order Ben should apply

1. Apply Phase 1 fixes first (tzinfo bug, hygiene cleanup) if not yet applied. The doc refresh assumes those are in place for the auto-gen pipeline to produce clean output.
2. Apply S5 (frontend HAIC brand hide) — pure UI change, quickest verification win.
3. Apply S1 + S2 + S4 (doc rewrites) — pure text, low risk.
4. Apply S3 (re-label pinned cases via `operator_review.py`), then regenerate the casebook via `submission_casebook.py`.
5. Re-run the auto-gen pipeline per S6 to refresh PACKET / READINESS / QUEUE / SET_BUILD.
6. Add `KNOWN_ISSUES.md` from S7 at the repo root.

## What this bundle intentionally does **not** do

- **Does not** invent an Entry B thesis statement beyond "SimSat-scoped Gemma-4 fine-tune" — team still owns thesis refinement with Guilherme.
- **Does not** edit auto-generated files directly (PACKET/READINESS/QUEUE/SET_BUILD) — next script run would overwrite our edits.
- **Does not** touch `LICENSE` — standard boilerplate, no session-driven change.
- **Does not** apply the Phase 1 tzinfo bug fix — already shipped in `review/2026-04-21-phase-1/`.
- **Does not** name "HAIC" in any pitch prose. The mechanism is described as "six non-compensatory viability gates" without brand reference.

## Phase 4 revision history

- Initial commits (3 commits, 12 files) — ship the bundle on `main`.
- Late-session amendment — lock in TTT thesis, finalize track labels (Liquid Track LFM2.5 / General AI Track Gemma-4), resolve HAIC into brand-hide + mechanism-feature stance. Amended files: this README, `S1-submission-brief/`, `S2-challenge-entry/`, `S7-known-issues-consolidated/`. Unchanged: `S3/`, `S4/`, `S5/`, `S6/`.
