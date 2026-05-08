# SimSat — Submission Abstract Drafts

> **Submitted by HumanAI Convention** · [humanaiconvention.com](https://humanaiconvention.com)

Form-fillable copy for the DPhi / Liquid AI hackathon submission. Each section is
self-contained — paste whichever block matches the prompt the form gives you.

---

## ELEVATOR (3 sentences, ~80 words)

SimSat is **on-orbit inference, continually refined by operator-labelled
JSON within uplink bandwidth parameters, gated by six non-compensatory
viability checks**. Operator labels are tokens, not gigabytes — a four-
character action choice fits the satellite's uplink budget where weight
updates can't, and the eight-key ObservationVLA contract makes the
feedback model-agnostic across any backend that emits it. Built,
fine-tuned, and exercised end-to-end on a real LFM2.5-VL-450M checkpoint
with seven runtime-adaptation receipts.

---

## ABSTRACT (3 paragraphs, ~250 words)

**Problem.** A satellite has no round trip to ground inside an encounter
window. The next window arrives in minutes; sensor distributions drift
between them. The on-board AI can't wait for ground retraining — but the
5 MB uplink that *is* available won't ferry weight updates either.
**What it can ferry is JSON.** The architectural question is: how do you
build an inference loop that gets reliably better from compact human
feedback without losing safety in the process?

**Approach.** SimSat answers with a structured eight-key JSON schema (the
ObservationVLA contract) that any vision-language backend can emit, and a
six-gate viability filter that governs which operator labels become
gradient signal. A WCLI trust layer scores each encounter against five
weighted features; a **two-scope test-time training loop** updates the
encoder LoRA *and* the planner head per pass under the same gates. We
trained the encoder LoRA on 165 operator-reviewed Sentinel-2 tiles via
`gallery_review.html`; the canonical v3 adapter ships at
[`HumanAIConvention/simsat-lfm25vl-450m-v3`](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3)
under Apache-2.0. The hybrid (LFM2.5-VL-450M tile encoder + MuZero
planner over encounter windows) fits the bandwidth budget end-to-end.

**Receipts.** v3 on a balanced 32-row matched-pair holdout:
`exact_action_agreement` 0.156 → **0.844** (+68.8 pp), `score_mae` 0.365
→ **0.055** (−31.0 pp). Three offline-tuning attempts to beat v3
(v4 +imbalanced data, v5 +balanced data, v3+ recipe variant) all
regressed — empirical evidence that further offline tuning is past the
inflection point. Seven runtime-adaptation receipts on a real
LFM2.5-VL checkpoint: 5/30/50-step stability runs (no divergence,
parse_rate 1.0 throughout), class-targeted runs that lift target-class
accuracy on a held-out probe (+37.5 pp on skip, +75 pp on defer with
action-token-weighted CE), and a stratified safety-floor receipt
(+3.1 pp net, no class catastrophically regressed) — the runtime
refinement claim, validated end-to-end.

---

## ONE-LINE PITCH (Twitter / form one-liners)

- **Cross-track:** SimSat is on-orbit inference, continually refined by
  low-bandwidth operator feedback through a JSON schema, gated by six
  non-compensatory viability checks — built, fine-tuned, and runtime-
  adaptation-validated end-to-end.
- **Liquid Track:** SimSat couples LFM2.5-VL with MuZero under six viability
  gates and two-scope TTT — the architecture handles drift at runtime, the
  +68.8 pp action-agreement adapter sets where drift starts, and seven
  TTT receipts demonstrate the runtime refinement loop on the real LFM
  checkpoint.
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
