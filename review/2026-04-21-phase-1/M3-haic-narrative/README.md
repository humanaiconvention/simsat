# M3 — HAIC-forward submission narrative

## The gap

The dashboard header renders `<span className="app-header-badge">HAIC Convention Layer</span>` (see `src/dashboard/frontend/src/App.tsx:48`), and the Architecture Explorer visualizes **8 HAIC variants (v3 through v8) alongside Liquid AI's LFM2-8b, LFM2-8b-a1b, Gemma 3/4, Llama 3.1, and Ministral**.

Yet the current `SUBMISSION_BRIEF.md` and `CHALLENGE_ENTRY.md` **do not mention HAIC once**.

Judges from Liquid AI will see a HAIC-heavy UI and a HAIC-silent pitch. This is the single highest-leverage submission change — bigger than any code fix — because:
1. It's specifically tuned to this judging panel (Liquid AI cares about the model landscape).
2. It reframes HAIC from "extra subsystem" to "the auditability moat above the planner."
3. It name-checks LFM2 in a way that shows you've studied Liquid's architecture.

## What's in this folder

| File | What it does |
|---|---|
| `SUBMISSION_BRIEF.md` | Full rewrite of the submission brief. Adds a **Convention Layer** section between **Claim** and **Evidence**. Updates **Runtime Truth** and **Demo Order** to reference the Architecture Explorer. |
| `CHALLENGE_ENTRY.md` | Full rewrite of the challenge entry. Adds two bullets to **What Is New** (HAIC convention layer, Architecture Explorer) and a new short **Convention Layer** section. |

## How to apply

These are full-file replacements. Compare against your current versions first:

```bash
cd D:\SimSat
# Sanity-check diffs before overwriting
diff SUBMISSION_BRIEF.md review\2026-04-21-phase-1\M3-haic-narrative\SUBMISSION_BRIEF.md
diff CHALLENGE_ENTRY.md review\2026-04-21-phase-1\M3-haic-narrative\CHALLENGE_ENTRY.md

# Copy when satisfied
copy review\2026-04-21-phase-1\M3-haic-narrative\SUBMISSION_BRIEF.md SUBMISSION_BRIEF.md
copy review\2026-04-21-phase-1\M3-haic-narrative\CHALLENGE_ENTRY.md CHALLENGE_ENTRY.md
```

## Key claims the rewrite makes

Each claim is grounded in code I reviewed, not speculation. If any feel off, revise before committing:

- **"Every stimulus bundle, every interview turn, every viability gate, and every PRISM entropy measurement is sealed into a Merkle-rooted receipt."** → grounded in `haic_test.py` sections 9-10 (PRISM + viability + entropy + 7-leaf Merkle receipt, 64-char hex root).
- **"The same Earth-imagery stream feeds both."** → grounded in `/haic/stimulus` accepting the same Sentinel + Mapbox payload the encounter pipeline uses.
- **"Today's interviewer runs against Anthropic, tomorrow's could run against LFM2 on-orbit."** → grounded in `requirements.txt` pinning `anthropic>=0.25.0` and the Architecture Explorer including `lfm2-8b_arch.html`. This frames provider-neutrality as a design goal, not a limitation.
- **"Architecture Explorer visualizing HAIC v3–v8 alongside LFM2-8b/LFM2-8b-a1b, Gemma 3/4, Llama 3.1, and Ministral."** → grounded in `src/dashboard/frontend/dist/arch/*.html`.

If any model visualization is aspirational rather than shipped, edit the wording before committing.
