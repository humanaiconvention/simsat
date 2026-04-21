# review/2026-04-21-phase-4 — Submission docs refresh + HAIC hide

Generated 2026-04-21, after the Phase 1/2/3 review phases and after a late-session pivot on the HAIC narrative.

## What this bundle is

Hand-edited updates to the submission-facing documents after the full-repo review completed earlier today. This bundle does **not** include auto-generated files (`SUBMISSION_PACKET.md`, `SUBMISSION_READINESS.md`, `REVIEW_QUEUE.md`, `REVIEW_SET_BUILD.md`) — those are produced by scripts and should be re-generated after the underlying fixes land locally. See `S6-autogen-regen/` for the regeneration checklist.

## Context — two decisions made today

1. **HAIC narrative: HIDE IT** (reverses the earlier same-day "feature it" call).
   Judges will see a clean encounter-planning + VLM pitch. HAIC stays in the repo for anyone who reads the source, but it comes out of the dashboard header and out of any submission-facing prose. The submission docs are already HAIC-silent — `S5-frontend-haic-hide/` covers the frontend changes needed to match.

2. **Two-track submission**: we are submitting to both the Liquid Track (LFM2.5-VL preferred, LFM2-VL fallback) and the General AI Track (new Gemma-4 fine-tune scoped to SimSat — **NOT** the v35-gov model from the separate HAIC × Gemma-4 Good hackathon). Thesis and model-choice finalization is still pending a Ben × Guilherme discussion; the doc updates here keep the two-track framing light-touch so nothing is over-committed.

## Contents

| Sub-bundle | What it touches | Kind |
|---|---|---|
| `S1-submission-brief/` | `SUBMISSION_BRIEF.md` | Doc rewrite |
| `S2-challenge-entry/` | `CHALLENGE_ENTRY.md` | Doc rewrite |
| `S3-casebook-notes/` | `SUBMISSION_CASEBOOK.md` + `operator_review.py` commands | Doc rewrite + script invocations |
| `S4-root-readme/` | top-level `README.md` | Patch description |
| `S5-frontend-haic-hide/` | `dashboard/frontend/src/App.tsx` | Frontend patch |
| `S6-autogen-regen/` | `submission_evidence.py`, `submission_readiness.py`, `review_queue_casebook.py` | Regeneration checklist |
| `S7-known-issues-consolidated/` | `KNOWN_ISSUES.md` (new file) | New consolidation |

## Order Ben should apply

1. Apply Phase 1 fixes first (tzinfo bug, hygiene cleanup) if not yet applied. The doc refresh depends on those being in place for the auto-gen pipeline to produce clean output.
2. Apply S5 (frontend HAIC hide) — pure UI change, quickest verification win.
3. Apply S1 + S2 + S4 (doc rewrites) — pure text, low risk.
4. Apply S3 (re-label pinned cases via `operator_review.py`), then regenerate the casebook via `submission_casebook.py`.
5. Re-run the auto-gen pipeline per S6 to refresh PACKET / READINESS / QUEUE / SET_BUILD.
6. Optionally add `KNOWN_ISSUES.md` from S7 at the repo root.

## What this bundle intentionally does **not** do

- **Does not** invent an Entry B thesis statement — that needs the team discussion.
- **Does not** edit auto-generated files directly (PACKET/READINESS/QUEUE/SET_BUILD) — next script run would overwrite our edits.
- **Does not** touch `LICENSE` — standard boilerplate, no session-driven change.
- **Does not** apply the Phase 1 tzinfo bug fix — already shipped in `review/2026-04-21-phase-1/`.
