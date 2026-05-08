# SimSat — Submission Abstract Drafts

Form-fillable copy for the DPhi / Liquid AI hackathon submission. Each section is
self-contained — paste whichever block matches the prompt the form gives you.

---

## ELEVATOR (3 sentences, ~80 words)

SimSat is an on-orbit satellite-tasking AI that handles distribution shift
**at runtime** — the next encounter window arrives in minutes, not hours, and
there is no round-trip budget for ground-side re-training. We pair an
LFM2.5-VL-450M tile encoder with a MuZero planner under a six-gate viability
filter that governs continual learning per encounter, and ship a fine-tuned
LoRA adapter that lifts holdout exact-action agreement from **0.156 to 0.844**
(+68.8 pp) on a balanced 32-row stratified holdout. The architecture is the
contribution; the model weights are evidence it works.

---

## ABSTRACT (3 paragraphs, ~250 words)

**Problem.** A satellite has no round trip to ground inside an encounter
window. Sensor distributions drift — clouds move, sun angle changes, target
seasons shift — and the AI on-board cannot wait minutes-to-hours for a human
to retrain and re-upload. The on-orbit constraint demands an architecture
that adapts in-pass without losing safety.

**Approach.** SimSat couples an LFM2.5-VL-450M (450M-param Liquid AI
vision-language model) tile encoder with a MuZero planner over encounter
windows. A WCLI trust layer scores each candidate window against five
weighted features and routes it through six non-compensatory viability gates
(weight drift, error bias, update rate, geometry, clarity, agreement). A
**two-scope test-time training loop** updates the encoder LoRA *and* the
planner head per pass under those gates. We trained the encoder LoRA on 165
operator-reviewed Sentinel-2 tiles; the canonical v3 adapter ships at
[`HumanAIConvention/simsat-lfm25vl-450m-v3`](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3)
under Apache-2.0.

**Result.** v3 on a balanced 32-row holdout (8 per action class):
`exact_action_agreement` 0.156 → **0.844** (+68.8 pp), `score_mae` 0.365 →
**0.055** (-31.0 pp), per-class accept **1.000** / refine **1.000** / defer
0.625 / skip 0.750. We additionally exercised the VLA-layer TTT loop
(`OnlineLoRAStepper`) end-to-end on the v1 adapter — 5/5 attempted online
gradient steps applied, 0 viability gates triggered, post-MAE held at perfect
on a stratified probe — closing the prior "wired, requires live stream" gap
on the runtime adaptation claim. We published a v4 negative result (-6.3 pp
action vs v3 on +20 imbalanced train rows) alongside the wins, because
honest negatives are part of the submission.

---

## ONE-LINE PITCH (Twitter / form one-liners)

- **Liquid Track:** SimSat couples LFM2.5-VL with MuZero under six viability
  gates and two-scope TTT — the architecture handles drift at runtime, the
  +68.8 pp action-agreement adapter sets where drift starts.
- **General AI Track:** SimSat fine-tunes Gemma-4-E2B for satellite encounter
  triage and wraps it in a model-agnostic scaffold with viability-gated TTT
  — any VLM emitting the eight-key ObservationVLA JSON contract inherits the
  same governed continual-learning loop.

---

## LINKS BLOCK (paste verbatim into form fields)

- **Code:** https://github.com/HumanAIConvention/SimSat
- **Canonical adapter (Liquid Track):** https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3
- **Reference adapter (Liquid Track v1, superseded):** https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v1
- **Canonical adapter (General AI Track):** https://huggingface.co/HumanAIConvention/simsat-gemma4-v11
- **Public training kernels (Kaggle):** `benhaslam/simsat-lfm2-5-vl-{v1,v3,v4,v5}-training`, `benhaslam/simsat-gemma4-v1-training`
- **Public training datasets (Kaggle):** `benhaslam/simsat-lfm-{v1,v2,v3,v5}`, `benhaslam/simsat-gemma4-v1`

---

## SUBMISSION CHECKLIST (judge-side reading order)

If the form has a free-text "where do I start" field:

> Read [`SUBMISSION_BRIEF.md`](./SUBMISSION_BRIEF.md) first (5 min — the rubric
> walk-through). Then [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md)
> (4 pinned operator-reviewed cases, one per scenario pack). Then
> [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md) for the long-form architectural
> thesis (TTT + viability gates + two-track design). Per-receipt depth in
> [`LFM_FINETUNE_METHODOLOGY.md`](./LFM_FINETUNE_METHODOLOGY.md) §6.5 (v3
> canonical numbers + recipe), [`LFM_TTT_POC.md`](./LFM_TTT_POC.md)
> §"Live receipt — 2026-05-07" (TTT proof-of-life), and
> [`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md) (v11 cross-distribution
> eval). [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) is the issue tracker — disclosed
> negatives + open gaps live there.

---

## TWO-TRACK MAPPING

If the form requires you to declare track-by-track:

**Liquid Track entry:**
- Use of Liquid Models (10pp): LFM2.5-VL-450M as the canonical encoder; v3 LoRA
  fine-tune (+68.8 pp action lift, public weights at
  HumanAIConvention/simsat-lfm25vl-450m-v3); VLA-layer TTT loop
  (OnlineLoRAStepper) exercised end-to-end with receipt artifact.
- Satellite Imagery (35pp): Sentinel-2 multi-spectral tiles, four scenario
  packs (maritime/disaster/urban-coastal/pedospheric). 268 operator-reviewed
  traces. Canonical Rotterdam case (urban-coastal, 48.78% cloud) demonstrates
  the calibration loop end-to-end.
- Technical Implementation (35pp): full FastAPI + dashboard stack runs locally
  via `python scripts/quickstart.py`. 209 unit tests pass. Apache-2.0 weights,
  AGPL-3 code.
- Demo (20pp): see attached video.

**General AI Track entry:**
- Satellite Imagery (20pp): same as Liquid Track.
- Liquid AI Models or APIs (25pp): same LFM2.5-VL encoder seat; SimSat is
  model-agnostic, so the adapter contract works for any VLM emitting the
  eight-key ObservationVLA JSON — including Liquid AI's API endpoints.
- Technical Implementation (35pp): Gemma-4-E2B v11 canonical adapter at
  HumanAIConvention/simsat-gemma4-v11 (410/410 LoRA tensor sanity gate PASS,
  exact agreement 0.86 in-distribution / 0.30 cross-distribution).
- Demo (20pp): see attached video.
