# S4 — Top-level `README.md` refresh

## What changed

Narrow, additive patch to the top-level `README.md`. The API/Dataset/Test Examples sections (lines ~92 onward in the current file) are unchanged — they describe upstream DPhi-Space contract and should stay verbatim.

Three edits in the challenge-entry section at the top:

1. Add a sentence pointing at the Two-Track Submission block in `CHALLENGE_ENTRY.md`.
2. Add a `Review Phases` subsection listing `review/2026-04-21-phase-1/` through `phase-4/`.
3. Add a `Known Issues` line pointing at `KNOWN_ISSUES.md` (from S7) or the phase-specific READMEs.

## Patch

Apply these edits to `D:\SimSat\README.md`:

### Edit 1 — After the first paragraph of "## Challenge Entry"

**Find:**

```markdown
This repo now also contains a challenge-focused encounter planner that treats observation opportunities as structured mission events instead of only continuous propagation. The challenge entry compares:
- a deterministic scaffold planner
- a WCLI-style trust-gated planner with an explicit `refine` action

The compact submission note and reproducible demo flow are in [CHALLENGE_ENTRY.md](/D:/SimSat/CHALLENGE_ENTRY.md).
The shortest judge-facing handoff is in [SUBMISSION_BRIEF.md](/D:/SimSat/SUBMISSION_BRIEF.md).
The current reviewed ObservationVLA evaluation is in [OBSERVATION_VLA_EVAL.md](/D:/SimSat/OBSERVATION_VLA_EVAL.md).
```

**Replace with:**

```markdown
This repo now also contains a challenge-focused encounter planner that treats observation opportunities as structured mission events instead of only continuous propagation. The challenge entry compares:
- a deterministic scaffold planner
- a WCLI-style trust-gated planner with an explicit `refine` action

The compact submission note and reproducible demo flow are in [CHALLENGE_ENTRY.md](/D:/SimSat/CHALLENGE_ENTRY.md).
The shortest judge-facing handoff is in [SUBMISSION_BRIEF.md](/D:/SimSat/SUBMISSION_BRIEF.md).
The current reviewed ObservationVLA evaluation is in [OBSERVATION_VLA_EVAL.md](/D:/SimSat/OBSERVATION_VLA_EVAL.md).

This repo is submitted to both tracks of the AI in Space hackathon — the Liquid Track (LFM2-VL / LFM2.5-VL backend) and the General AI Track (SimSat-specific Gemma-4 fine-tune). See the Two-Track Submission section of [CHALLENGE_ENTRY.md](/D:/SimSat/CHALLENGE_ENTRY.md) for per-track model details.
```

### Edit 2 — Add a new subsection right before `## Upcoming Hackathon: AI in Space | Liquid AI x DPhi Space`

**Insert:**

```markdown
### Review Phases

The 2026-04-21 full-repo code review is preserved in the repo so judges can audit the work we found and shipped:

- `review/2026-04-21-phase-1/` — Tier-1 correctness fixes (tzinfo bug, hygiene). **Apply first.**
- `review/2026-04-21-phase-2/` — Tier-2 thesis/calibration work (accept→refine sweep, MAE calibration reframe, re-materialization scripts).
- `review/2026-04-21-phase-3/` — Entry B backend scaffold (`entry-b/backend` branch) and Google Drive recon block.
- `review/2026-04-21-phase-4/` — Submission-doc refresh + frontend HAIC-hide patch.

### Known Issues

See `KNOWN_ISSUES.md` at the repo root for the consolidated list with status per item. Short version: the Phase 1 tzinfo bug in `src/sim/simulator.py:111` is the only known Tier-1 correctness issue; the rest are rigor/polish items disclosed in the submission docs rather than hidden.

```

## Apply

Use `Edit` or manual find-and-replace against `D:\SimSat\README.md`. The full-file replacement is **not** provided here because 19 KB of upstream API / dataset docs would be duplicated verbatim and risk drift if upstream changes them.

## Verify

```bash
grep -c "Two-Track" README.md             # expect 1
grep -c "Review Phases" README.md         # expect 1
grep -c "Known Issues" README.md          # expect 1
grep -c "HAIC" README.md                  # expect 0
```
