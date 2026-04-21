# S1 — `SUBMISSION_BRIEF.md` refresh

## What changed

- Added a **Tracks** block noting both-track submission. Kept the thesis statement unchanged; does not invent an Entry B thesis.
- Added an **Evidence honesty** line acknowledging that ObservationVLA's reviewed eval is action-agreement strong (1.00) but magnitude under-confident (MAE 0.27 on usefulness score).
- Added a **Known Issues** pointer directing readers to `review/2026-04-21-phase-1/`, `phase-2/`, `phase-3/`, and the new `KNOWN_ISSUES.md` (if applied from S7).
- Preserved all existing content — demo order, reviewed cases table, reproduce commands — unchanged.

## What did **not** change

- No HAIC mentions added. The brief was HAIC-silent already, which aligns with today's "hide HAIC" decision. Nothing to remove.
- No new thesis wording for Entry A vs Entry B. That is deferred pending team discussion per the project document.

## Apply

Drop `SUBMISSION_BRIEF.md` from this directory over `D:\SimSat\SUBMISSION_BRIEF.md` (or the repo root equivalent).

## Verify

```bash
grep -c "HAIC" SUBMISSION_BRIEF.md     # expect 0
grep -c "Tracks" SUBMISSION_BRIEF.md   # expect 1
grep -c "MAE" SUBMISSION_BRIEF.md      # expect 1
```
