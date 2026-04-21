# M4 — Rewrite the 3 pinned review notes

## The problem

All three pinned submission cases share a review-note template that sounds like this:

> "No observable cloud cover, user wouldn't know Houston but appears correct and useful."

The phrasing "user wouldn't know X" undermines the reviewer before the sentence ends. Judges reading this will reasonably ask: if the reviewer can't recognize the target, how do they know it's correct?

That's not the intent — the intent is reviewer honesty about the limits of visual ground-truthing. The fix is to replace "I wouldn't know X" with the **observable features** that made the decision (cloud %, Sentinel source, shape, infrastructure patterns) while keeping the honesty.

## What this folder contains

`SUBMISSION_CASEBOOK.md` — a full-file replacement with rewritten review notes for all three pinned cases.

Each new review note:
1. Opens with **observable metadata** (Sentinel source, cloud %).
2. Cites **visual features** consistent with the target identity.
3. Explains **why the reviewer accepted** on available evidence.
4. Includes a **bracketed reviewer note** ("verify X against cached image before finalizing") that preserves epistemic honesty without undermining the decision.

## ⚠️ Verification caveat (important)

These rewrites are grounded in the **metadata only** (cloud %, Sentinel source, scenario pack, cloud cover values from the existing casebook) because I didn't see the actual PNG images from my environment. The descriptions of visible features are plausible inferences, not confirmed observations.

**Before committing this file to `SUBMISSION_CASEBOOK.md`, open each of the three pinned images and verify the descriptions match what you see:**

- `D:\SimSat\submission_assets\disaster_response_weather_houston_ship_channel.png`
- `D:\SimSat\submission_assets\maritime_chokepoints_suez_canal.png`
- `D:\SimSat\submission_assets\urban_coastal_ambiguity_san_francisco_bay.png`

If any described feature isn't actually visible (e.g., "ship wakes" at Houston), edit or remove that phrase. It is better to commit a shorter, accurate description than a confident but wrong one.

## How to apply

```bash
cd D:\SimSat
# Review the replacement first
diff SUBMISSION_CASEBOOK.md review\2026-04-21-phase-1\M4-review-notes\SUBMISSION_CASEBOOK.md

# After verifying against the actual images, overwrite
copy review\2026-04-21-phase-1\M4-review-notes\SUBMISSION_CASEBOOK.md SUBMISSION_CASEBOOK.md
```

## Related Phase 2 item

**S3** (Should-Do) will re-materialize these 3 pinned cases under the `clip_local` backend so the casebook stops showing `Observation runtime: stub`. After S3, the review notes in this file may want a small addendum like *"Re-assessed under `clip_local` backend after initial `stub` labeling."*
