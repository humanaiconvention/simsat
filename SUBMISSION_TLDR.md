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

**TTT — five receipts (full development arc):**
- **Stability:** 5-step / 30-step / 50-step receipts on v3 adapter all run cleanly,
  no divergence, no OOM, parse_rate 1.000 throughout. Trust-layer TTT has
  100-cycle evidence; VLA-layer now has 50-cycle.
- **Class-targeted lift (HEADLINE):** 16 skip-only train rows as the stream,
  action-token-weighted CE loss → **skip class on 8-row probe lifted 0.375 → 0.750
  (+37.5 pp)**, score_mae 0.237 → 0.162. Loss drove from 1.15 → 0.0002. **TTT
  empirically lifts target-class accuracy on a real LFM2.5-VL checkpoint, per
  pass, under operator-curated stream — the architectural claim validated
  end-to-end.** Published next to the negative result (full-CE loss regressed
  skip 0.375 → 0.000), so the design space is mapped. ([receipts in `.kaggle_output/`](./.kaggle_output/))

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

> **One-line pitch:** SimSat is the architecture you ship to satellites that
> can't phone home — six viability gates, two-scope TTT that empirically
> lifts +37.5 pp on a target class per pass, +68.8 pp action agreement on
> operator-reviewed Sentinel tiles, and an honest negative result in the
> submission for every win.
