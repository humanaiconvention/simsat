# SimSat — One-Page TL;DR for Judges

**Hackathon:** AI in Space (DPhi Space × Liquid AI), May 2026.
**Tracks submitted:** Liquid Track + General AI Track.
**Code:** [`HumanAIConvention/SimSat`](https://github.com/HumanAIConvention/SimSat) (AGPL-3) · **Weights:** [`HumanAIConvention`](https://huggingface.co/HumanAIConvention) on HF (Apache-2.0)

## What we built

An on-orbit satellite-tasking AI for Sentinel-2 encounter triage. It treats
each window as a small decision: `accept · refine · defer · skip`. The
architecture handles **distribution shift at runtime** — no ground-side
re-training between encounter windows — under **six non-compensatory
viability gates** that govern continual learning per pass.

## The single claim

A satellite has **minutes, not hours**, between windows. There's no
round-trip budget for ground-side retraining. The whole system is built
around that constraint: tile encoder + planner that fit a 5 MB uplink,
viability gates that act as the human-in-the-loop substitute, and
two-scope test-time training that actually moves both the encoder LoRA
and the trust-layer weights *during the pass*, not after it.

## Headline numbers

**Liquid Track — LFM2.5-VL-450M v3 LoRA fine-tune** ([adapter](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3)):

| Metric | Base | Tuned (v3) | Δ |
|---|---|---|---|
| `exact_action_agreement` | 0.156 | **0.844** | **+68.8 pp** |
| `score_mae` (lower better) | 0.365 | **0.055** | **−31.0 pp** |
| Per-class accept / refine / defer / skip | — | **1.000 / 1.000** / 0.625 / 0.750 | balanced |

Matched-pair eval, 32-row stratified holdout, 8 per action class. Same holdout
across v1 / v3 / v4 / v5 for direct comparison.

**TTT proof-of-life** ([receipt](.kaggle_output/ttt_proof_of_life_receipt.json)):
5/5 attempted online-LoRA gradient steps applied, 0 viability gates triggered,
post-MAE held perfect on stratified probe, `lora_delta_l2` grew monotonically
0.0008 → 0.0021. Mechanism is real, not vapor.

**General AI Track — Gemma-4-E2B v11** ([adapter](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11)):
in-distribution N=37 exact 0.86 / useful 0.97 / MAE 0.13 (vs always-majority
baseline 0.54, random 0.25). 410/410 LoRA tensor sanity gate pass
([V11_AUDIT.md](./V11_AUDIT.md)).

## Honest negatives published next to wins

- **v4 fine-tune regressed −6.3 pp** on action vs v3 (imbalanced +20 train);
  **v5 regressed −15.6 pp** (defer-class-balanced +56 train cratered skip
  0.75→0.25). Kept v3 canonical both times. **Two consecutive negative
  results on +data confirm v3 is at the data-curve inflection point** —
  the architectural argument for runtime TTT over more offline corpus
  expansion. ([§6.6 / §6.7 methodology](./LFM_FINETUNE_METHODOLOGY.md))
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

> **One-line pitch:** SimSat is the architecture you ship to satellites that
> can't phone home — six viability gates, two-scope TTT, +68.8 pp action
> agreement on operator-reviewed Sentinel tiles, and an honest negative
> result in the submission for every win.
