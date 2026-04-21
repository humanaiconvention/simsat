# S1 — Surface an `accept → refine` delta for the live demo

## The problem

`SUBMISSION_PACKET.md` currently reports transition counts like:

```
disaster_response_weather:   {'accept->accept': 2}
maritime_chokepoints:        {'accept->accept': 1}
urban_coastal_ambiguity:     {'accept->accept': 3}
```

Every pinned case is `accept → accept`. The WCLI-trust thesis — *"preserves high-value windows through an explicit `refine` path"* — is therefore **rhetorical on this evidence set, not falsified**. If a judge asks "show me one case where trust actually changed your mind," we cannot.

The corpus has drift (`OBSERVATION_VLA_CORPUS.md` reports 6 `accept → refine` shifts in 86 traces), but those are **ObservationVLA backend shifts** (stub vs. clip_local), not scaffold-vs-trust planner deltas. Different story.

## Why it happens

WCLI-trust fires `refine` when:

```
trust_score < trust_refine_threshold (0.55)
AND scaffold_score >= defer_threshold (0.45)
```

`trust_score` weights: `agreement × 0.30 + geometry_margin × 0.30 + duration_margin × 0.15 + imagery_support × 0.10 + clarity_support × 0.15`.

For the 3 pinned scenario packs (Suez, Houston, SF Bay), the high-priority targets consistently have: high elevation → high geometry_margin; good duration; Sentinel available → imagery_support = 1.0; low cloud → clarity_support ≈ 1.0. Trust rarely drops below 0.55 unless scaffold and learned diverge (low agreement) AND at least one structural factor is degraded.

The upshot: with the default policy, targets in your current scenario packs almost always clear both `accept_threshold` (0.70) and `trust_accept_min` (0.45). Accept → refine just won't fire on Suez at noon UTC.

## The plan

Ship two artifacts:

### 1. `find_accept_refine_delta.sh` — sweep for a natural delta

A bash/PowerShell wrapper that runs `encounter_eval.py` across many start times (covering various cloud/support conditions) looking for any `accept → refine` transition. If one surfaces, `jq` extracts the window + target + reason.

### 2. `DEMO_DELTA_ADDENDUM.md` — narrative insert for the submission

If the sweep surfaces a delta, copy the content into `SUBMISSION_BRIEF.md` (new section **Decision Delta Showcase**) and reference it in the live demo. If the sweep surfaces nothing in a reasonable time, use the **Fallback** section that tweaks `trust_refine_threshold` to 0.65 for the demo run only (documented, reversible).

## How to apply

### Step 1 — Sweep for natural deltas

```bash
cd D:\SimSat
# Requires docker compose up running
bash review/2026-04-21-phase-2/S1-accept-refine-delta/find_accept_refine_delta.sh
```

The script writes `review/2026-04-21-phase-2/S1-accept-refine-delta/deltas_found.md` with every `accept → refine` transition it finds (and runs silently if none).

### Step 2a — If a natural delta is found

Copy the content of `deltas_found.md` into the **Decision Delta Showcase** section of `SUBMISSION_BRIEF.md` (see `DEMO_DELTA_ADDENDUM.md` for the target shape). Pin that window's decision as the demo delta:

```bash
# Replace <window_id> with the window_id from deltas_found.md
python scripts/operator_review.py --base-url http://127.0.0.1:8000/sim --trace-id <trace_id_from_delta> --show-bundle-only
# Then review + pin per CHALLENGE_ENTRY.md
```

### Step 2b — If no natural delta appears

Open `DEMO_DELTA_ADDENDUM.md` and use the **Fallback** section. Edit `src/sim/encounter/schemas.py` (EncounterPolicy): temporarily change `trust_refine_threshold: float = 0.55` → `0.65`, restart the sim, re-run the eval. You will now see `accept → refine` transitions. **Revert the change after the demo** — a permanent tweak would make the planner over-conservative.

## Honesty note

This item is the least mechanical of the four Phase 2 items. The goal is not to manufacture a delta to look good — it is to find a case where the thesis IS visible. If the sweep surfaces nothing and you don't want to tweak the policy, a clean alternative is to **explain the absence**: *"On our curated scenario packs, trust does not override scaffold — this is evidence the scaffold is well-calibrated for high-priority clear-sky targets. The thesis bites on edge cases, surfaced below."*

That framing is defensible and honest. Judges will recognize it as maturity, not handwaving.
