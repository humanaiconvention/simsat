# S2 — `CHALLENGE_ENTRY.md` refresh

## What changed

- Added a **Two-Track Submission** section near the top naming Entry A (Liquid) and Entry B (General AI) with per-track model choices. Does not invent new thesis wording — team discussion is still pending.
- Added a **Known Issues and Review Bundles** section near the bottom pointing readers at `review/2026-04-21-phase-1/` through `phase-4/`. Gives judges a transparent audit trail instead of hiding the review work.
- Preserved the existing thesis, scenario packs, evaluation outputs, demo flow, reproduce commands, and submission framing verbatim. This is an **additive** refresh.

## What did **not** change

- No HAIC mentions added. The challenge entry was HAIC-silent already and stays that way.
- The existing `Submission Framing` paragraph at the bottom is unchanged — that is the canonical pitch wording.

## Apply

Drop `CHALLENGE_ENTRY.md` from this directory over `D:\SimSat\CHALLENGE_ENTRY.md`.

## Verify

```bash
grep -c "HAIC" CHALLENGE_ENTRY.md               # expect 0
grep -c "Two-Track" CHALLENGE_ENTRY.md          # expect 1
grep -c "Known Issues" CHALLENGE_ENTRY.md       # expect 1
grep -c "v35-gov" CHALLENGE_ENTRY.md            # expect 1 (the explicit disclaimer)
```
