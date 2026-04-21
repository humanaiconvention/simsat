# S4 — Architecture Explorer subsection for CHALLENGE_ENTRY.md

## Why

Phase 1 (M3) added the **Convention Layer** section and name-dropped LFM2-8b. S4 finishes the job: a dedicated subsection that **lists every model visualized**, explains why each is there, and makes the comparison narrative legible without opening the dashboard.

This is what turns "we have an Architecture Explorer" into "here is the model landscape and here is why it matters."

## What's in this folder

`CHALLENGE_ENTRY_architecture_explorer_section.md` — the new subsection, ready to paste into `CHALLENGE_ENTRY.md`.

## Where to paste it

In `CHALLENGE_ENTRY.md` (the Phase 1 M3 rewrite), insert this subsection **immediately after the `## Convention Layer` section and before `## Scenario Packs`**.

## Verification caveat

This subsection makes claims about the architectural character of each model family. I grounded those in the public technical reports (LFM2 activation sparsity, Gemma 4 family, Llama 3.1 8B, Ministral-3B) and the naming pattern in your `dist/arch/*.html` files (haic-v3-2b through haic-v8-2b, with q2/q3/q4 quantization variants on haic-v3-2b).

**Before committing, verify:**
1. The HAIC v3–v8 line about "iterative refinements under the convention framing" matches what each architecture file actually documents. If specific HAIC versions added/removed specific components, note them.
2. `haic-v3-8b` exists as a larger sibling of `haic-v3-2b` (the filename suggests so). If not, remove that bullet.
3. The quantization-ablation claim (`q2/q3/q4` on `haic-v3-2b`) is a reasonable reading of the filenames. If those are something else (e.g., different data mixes), update the phrasing.
4. Anything described as "comparison" should actually be visualized in the Explorer. If any `.html` file is a stub, remove or mark it as "aspirational / planned."
