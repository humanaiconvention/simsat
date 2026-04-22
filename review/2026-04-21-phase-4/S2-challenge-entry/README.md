# S2 — `CHALLENGE_ENTRY.md` refresh (amended)

## What changed (amended)

- **Thesis rewritten** to name SimSat as "a governed on-orbit continual-learning loop" and to position TTT + viability gates as the central architectural claim.
- **New "Why this must run in orbit" section** with the three-layer argument: distribution shift without ground-truth validator (six non-compensatory viability gates), bandwidth, latency. Lead argument is the viability-gate mechanism — our unique differentiator.
- **New "Test-Time Training — Stacked" section** explaining VLA-layer TTT, trust-layer TTT, and the common viability-gate filter. Makes the architectural claim explicit: stacked TTT is safe only because the gates screen adaptations.
- **Two-Track Submission section tightened** — "Liquid Track (LFM2.5)" and "General AI Track (Gemma-4)" as canonical labels, LFM2-VL fallback language dropped.
- **"What Is New" list updated** to include stacked TTT and six viability gates as first-class novelty claims.
- **Known Issues and Review Bundles section retained** with Phase 4 pointer updated to reflect the amended content.
- Preserved everything else verbatim — scenario packs, evaluation outputs, judge-facing scorecard, submission evidence commands, reproducible demo, low-compute rehearsal, suggested live demo flow, submission framing paragraph.

The entry does **not** name "HAIC" anywhere. The viability mechanism is described mechanically (entropy reduction, extraction risk, PRISM consistency, participation covenant, federated exchange, epistemic alignment) without brand reference.

## What did **not** change

- No HAIC mentions. Already HAIC-silent after the first refresh; the mechanism-feature amendment keeps the brand out.
- The canonical `Submission Framing` paragraph at the bottom is updated minimally to reflect TTT and viability gates in the one-paragraph pitch wording.

## Apply

Drop `CHALLENGE_ENTRY.md` from this directory over `D:\SimSat\CHALLENGE_ENTRY.md`.

## Verify

```bash
grep -c "HAIC" CHALLENGE_ENTRY.md                          # expect 0
grep -c "Test-Time Training\|TTT" CHALLENGE_ENTRY.md       # expect >=2 (section + mentions)
grep -c "viability" CHALLENGE_ENTRY.md                     # expect >=3 (section + mentions)
grep -c "entropy reduction\|extraction risk\|PRISM\|participation covenant\|federated exchange\|epistemic alignment" CHALLENGE_ENTRY.md  # expect 6 (the six gates listed)
grep -c "LFM2.5" CHALLENGE_ENTRY.md                        # expect >=1 (Liquid Track)
grep -c "Gemma-4" CHALLENGE_ENTRY.md                       # expect >=2 (General AI Track + v35-gov disambiguation)
grep -c "LFM2-VL" CHALLENGE_ENTRY.md                       # expect 0 (fallback language removed)
grep -c "v35-gov" CHALLENGE_ENTRY.md                       # expect 1 (explicit disclaimer)
grep -c "Two-Track" CHALLENGE_ENTRY.md                     # expect 1
grep -c "Known Issues" CHALLENGE_ENTRY.md                  # expect 1
```
