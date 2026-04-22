# S1 — `SUBMISSION_BRIEF.md` refresh (amended)

## What changed (amended)

- **New "Why this must run in orbit" section** leads the brief with the three-layer orbital argument: distribution shift without a ground-truth validator (six non-compensatory viability gates), bandwidth (5 MB up / 10 MB down), latency (in-pass adaptation before the next encounter window). The viability-gate argument is our unique differentiator and leads.
- **Claim section expanded** to name TTT as the central technical thesis — stacked at VLA layer and trust layer, both gated by viability.
- **Tracks section refined** — "Liquid Track (LFM2.5)" and "General AI Track (Gemma-4)" as canonical labels. Drops the earlier "LFM2-VL fallback" language.
- **Runtime Truth block expanded** to disclose which TTT streams are wired vs. documented, and to reaffirm the MAE 0.27 honesty line.
- **Known Issues pointer retained** — directs readers to `review/2026-04-21-phase-1/` through `phase-4/` and the consolidated `KNOWN_ISSUES.md`.

The brief does **not** name "HAIC" anywhere. The viability mechanism is described mechanically; the brand lives only in repo source.

## What did **not** change

- Reviewed cases table (Suez 0.95, Houston 0.92, SF Bay 0.90) — facts, unchanged.
- Demo order — same three-doc walk (PACKET → CASEBOOK → CHALLENGE_ENTRY).
- Reproduce commands — same three scripts.

## Apply

Drop `SUBMISSION_BRIEF.md` from this directory over `D:\SimSat\SUBMISSION_BRIEF.md` (or the repo root equivalent).

## Verify

```bash
grep -c "HAIC" SUBMISSION_BRIEF.md                    # expect 0
grep -c "viability" SUBMISSION_BRIEF.md               # expect >=2 (lead section + claim)
grep -c "Test-Time Training\|TTT" SUBMISSION_BRIEF.md # expect >=1
grep -c "LFM2.5" SUBMISSION_BRIEF.md                  # expect 1 (Liquid Track line)
grep -c "Gemma-4" SUBMISSION_BRIEF.md                 # expect 1 (General AI Track line)
grep -c "LFM2-VL" SUBMISSION_BRIEF.md                 # expect 0 (fallback language removed)
grep -c "v35-gov" SUBMISSION_BRIEF.md                 # expect 1 (the explicit disclaimer)
grep -c "MAE" SUBMISSION_BRIEF.md                     # expect 1
```
