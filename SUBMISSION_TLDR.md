# SimSat — One-Page TL;DR for Judges

**Hackathon:** AI in Space (DPhi Space × Liquid AI), May 2026.
**Tracks submitted:** Liquid Track + General AI Track.
**Code:** [`HumanAIConvention/SimSat`](https://github.com/HumanAIConvention/SimSat) (AGPL-3) · **Weights:** [`HumanAIConvention`](https://huggingface.co/HumanAIConvention) on HF (Apache-2.0)

## What we built

**On-orbit inference, continually refined by operator-labelled JSON
within uplink bandwidth parameters, gated by six non-compensatory
viability checks.**

Sentinel-2 encounter triage treats each window as one of four operator-
meaningful decisions (`accept · refine · defer · skip`). The model runs
on-board the satellite; operator labels come back up the 5 MB uplink as
compact JSON tokens (not gigabytes of weight updates); a six-gate
viability filter governs which labels become gradient signal; an
eight-key ObservationVLA contract makes the entire feedback loop
model-agnostic.

## The single claim

A satellite has **minutes, not hours**, between encounter windows.
The 5 MB uplink budget can't ferry weight updates between them. **What
it can ferry is JSON.** SimSat is the architecture that turns that
constraint into a feature: small structured human feedback per pass,
six viability gates filtering which feedback becomes gradient signal,
two-scope test-time training adapting both the encoder LoRA and the
trust-layer weights in flight under those gates.

## Headline numbers

**Liquid Track — LFM2.5-VL-450M v3 LoRA fine-tune** ([adapter](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3)):

| Metric | Base | Tuned (v3) | Δ |
|---|---|---|---|
| `exact_action_agreement` | 0.156 | **0.844** | **+68.8 pp** |
| `score_mae` (lower better) | 0.365 | **0.055** | **−31.0 pp** |
| Per-class accept / refine / defer / skip | — | **1.000 / 1.000** / 0.625 / 0.750 | balanced |

Matched-pair eval, 32-row stratified holdout, 8 per action class. Same holdout
across v1 / v3 / v4 / v5 for direct comparison.

**TTT — seven receipts (full development arc):**
- **Stability:** 5-step / 30-step / 50-step receipts on v3 adapter all run cleanly,
  no divergence, no OOM, parse_rate 1.000 throughout. Trust-layer TTT has
  100-cycle evidence; VLA-layer now has 50-cycle.
- **Class-targeted lift (HEADLINE — two classes):** 16 class-only train rows
  as the stream, action-token-weighted CE loss, on the canonical v3 adapter:
    - **Skip class:** 0.375 → 0.750 (+37.5 pp on the 8-row held-out probe)
    - **Defer class:** 0.125 → 0.875 (+75.0 pp on the 8-row held-out probe)
    - Average lift: **+56.25 pp**. Loss drove ~1.0 → ~0 in both runs.
  **TTT empirically lifts target-class accuracy on a real LFM2.5-VL
  checkpoint, per pass, under operator-curated stream — the architectural
  claim validated end-to-end on TWO independent classes.** Published next
  to the v1 negative result (full-CE loss regressed skip 0.375 → 0.000), so
  the design space is mapped: action-token-weighted loss is the necessary
  ingredient.
- **Robustness check (stratified TTT, full 32-row holdout):** balanced
  4-per-class operator-feedback stream over 16 steps, **net +3.1 pp overall
  with NO catastrophic regression** — skip lifted +37.5 pp, defer/refine
  each dropped one sample, accept held perfect. The single-class runs are
  the upper bound; stratified is the safety floor.
  ([receipts in `.kaggle_output/`](./.kaggle_output/))

**General AI Track — Gemma-4-E2B v11** ([adapter](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11)):
in-distribution N=37 exact 0.86 / useful 0.97 / MAE 0.13 (vs always-majority
baseline 0.54, random 0.25). 410/410 LoRA tensor sanity gate pass
([V11_AUDIT.md](./V11_AUDIT.md)).

## Honest negatives published next to wins

- **Three consecutive offline-tuning attempts regressed vs v3** — v4 (+20
  imbalanced data, −6.3 pp), v5 (+56 class-balanced data, −15.6 pp + skip
  0.75→0.25), v3+ (recipe variant on same v2 dataset, −3.1 pp). v3 sits at
  a local optimum that further offline tuning cannot escape. **Empirical
  evidence (three independent lines: data-imbalanced, data-balanced, recipe)
  for the runtime-TTT lane**. None of v4/v5/v3+ promoted to HF; v3 stays
  canonical. ([§6.6 / §6.7 / §6.8 methodology](./LFM_FINETUNE_METHODOLOGY.md))
- **v12 retrain on dataset v4 hit a parse-rate regression** vs v11 — kept v11
  canonical, published the negative. ([KNOWN_ISSUES.md #28](./KNOWN_ISSUES.md))
- **Cross-distribution N=152 v11 eval = 0.30 vs in-distribution N=37 = 0.86**
  — the **56-point gap is the architectural argument**, not a hidden failure.
  ([OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md))

## What the prize hardware enables

Wired but not benchmarked yet (requires a live encounter stream the
**NVIDIA Orin 16 GB** prize hardware unlocks):
- Long-horizon two-scope TTT (encoder LoRA + MuZero head jointly), per-pass.
- Spectral-register tile expansion in the MuZero eval corpus.
- Direct on-Orin latency measurement (currently RTX 2080: 69 ms/tile p50,
  78 ms p95 — sub-250 ms Liquid AI claim verified on local hardware).

## Reading order for judges

| Want… | Read |
|---|---|
| 30-second pitch | this file |
| Rubric mapping | [`SUBMISSION_BRIEF.md`](./SUBMISSION_BRIEF.md) (250 lines) |
| 4 pinned operator-reviewed cases | [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md) |
| Architectural thesis | [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md) |
| LFM fine-tune recipe + holdout | [`LFM_FINETUNE_METHODOLOGY.md`](./LFM_FINETUNE_METHODOLOGY.md) |
| Run-time TTT receipt | [`LFM_TTT_POC.md`](./LFM_TTT_POC.md) |
| All disclosed gaps | [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) |
| 4-min walkthrough | demo video (attached to submission) |

## Submission integrity

- 209 unit tests + reviewed-case eval pass on Linux CI ([`tests.yml`](./.github/workflows/tests.yml)).
- LFM v1, v3, v4, v5 training kernels all public on Kaggle (`benhaslam/simsat-lfm2-5-vl-{v1,v3,v4,v5}-training`).
- v1, v3 LoRA adapters published with documented recipes on HuggingFace.
- Apache-2.0 weights / AGPL-3 code.

---

> **One-line pitch:** SimSat is on-orbit inference, continually refined by
> operator-labelled JSON within uplink bandwidth parameters — six viability
> gates govern which feedback becomes gradient signal, and seven runtime-
> adaptation receipts on a real LFM2.5-VL checkpoint demonstrate the loop
> end-to-end (mechanism + stability + class-targeted lift + balanced
> safety floor).
