# SimSat — Liquid Track Entry

> **Submitted by HumanAI Convention** · [humanaiconvention.com](https://humanaiconvention.com)
>
> SimSat operationalizes the **Viability Convention** hypothesis: that AI
> systems must balance synthetic data with human lived experience, governed
> by an ethical framework that filters which feedback becomes gradient
> signal. *The architecture is the contribution; the satellite is the test
> case.*

**Audience:** judges scoring the Liquid Track of the AI in Space Hackathon
(DPhi Space × Liquid AI). Rubric: 10 / 35 / 35 / 20 (Use of Liquid Models /
Innovation & Problem-Solution Fit / Technical Implementation / Demo).

**Same repo, same code as the General AI Track entry** — see
[`GENERAL_AI_TRACK.md`](./GENERAL_AI_TRACK.md). The architectural contribution
is shared; what differs is which model backend fills the encoder seat.

---

## What this submission delivers for the Liquid Track

### 1. Use of Liquid Models — 10%

**Liquid AI's LFM2.5-VL-450M is the canonical encoder, fine-tuned end-to-end:**

- **v3 LoRA adapter on Hugging Face:**
  [`HumanAIConvention/simsat-lfm25vl-450m-v3`](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3)
  (Apache-2.0, 17.9 MB safetensors, full README with recipe + holdout
  numbers).
- **Holdout eval (matched-pair, 32-row stratified, 8 per action class):**

| Metric | Base | Tuned (v3) | Δ |
|---|---|---|---|
| `exact_action_agreement` | 0.156 | **0.844** | **+68.8 pp** |
| `score_mae` (lower better) | 0.365 | **0.055** | **−31.0 pp** |
| Per-class accept / refine / defer / skip | — | **1.000 / 1.000 / 0.625 / 0.750** | balanced |

- **Public training kernels on Kaggle:** `benhaslam/simsat-lfm2-5-vl-{v1,v3,v4,v5}-training`
- **VLA-layer TTT** (`OnlineLoRAStepper` in [`src/sim/observation_vla/lfm_ttt.py`](./src/sim/observation_vla/lfm_ttt.py))
  exercised end-to-end on the v3 adapter — see "TTT receipts" below.
- **Encoder latency (RTX 2080):** 69 ms/tile mean, 78 ms p95 (LFM2.5-VL-450M).
  1.6B variant: 246 ms p95 — matches Liquid AI's "sub-250 ms edge"
  benchmark.

**Recipe + protocol:** [`LFM_FINETUNE_METHODOLOGY.md`](./LFM_FINETUNE_METHODOLOGY.md)
(target_modules selected verbatim from `Liquid4All/leap-finetune`
`DEFAULT_VLM_LORA`, stratified 8-per-class hold-out, three concentric
sanity gates, matched-pair eval, disclosed limitations).

### 2. Innovation & Problem-Solution Fit — 35%

**Architectural contribution: stacked TTT under six non-compensatory
viability gates.** Detailed in [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md);
the one-paragraph thesis:

> A satellite has minutes, not hours, between encounter windows. There's
> no round-trip budget for ground retraining. SimSat couples the LFM2.5-VL-450M
> tile encoder with a MuZero planner over encounter windows — the hybrid
> fits a 5 MB satellite uplink budget while two-scope test-time training
> adapts both the encoder LoRA and the planner head per pass under the
> same six viability gates that govern the trust layer's online updates.
> The viability gates are the substitute for human-in-the-loop on a
> satellite — every accepted update carries quality-verified, magnitude-
> bounded, bias-checked information about the world before touching model
> state.

**Distribution-shift demo (trust-layer 10-seed evidence,
[`ttt_stability_analysis.md`](./ttt_stability_analysis.md)):** coastal →
polar regime shift, polar MAE 0.161 → 0.150 (−6.9 %, **94% of the
polar-regime improvement in one cycle of operator feedback**). Weight
re-adaptation: geometry +0.061, clarity −0.109. Concrete evidence the
on-orbit loop tracks regime shifts, not just plateaus on fixed data.

### 3. Technical Implementation — 35%

This is where the LFM-specific evidence lives.

**Three offline-fine-tune attempts to beat v3 — all regressed** (published
honestly):
- v4 (+20 imbalanced refine/accept rows, 185 train): **−6.3 pp action**
- v5 (+56 defer-class-balanced rows, 241 train): **−15.6 pp action**, skip 0.75 → 0.25
- v3+ (recipe variant on same v2 dataset: lr=1e-4, 8 epochs, dropout=0.10): **−3.1 pp action**

**Three independent angles, same conclusion: v3 sits at a local optimum
on this architecture × data × holdout combination that further offline
tuning cannot escape.** Empirical evidence (not just claim) for the
runtime-TTT lane being the next investment. None promoted to HuggingFace;
v3 stays canonical. Detail in [`LFM_FINETUNE_METHODOLOGY.md`](./LFM_FINETUNE_METHODOLOGY.md)
§6.6 / §6.7 / §6.8.

**Seven TTT receipts on a real LFM2.5-VL-450M checkpoint** — full
development arc documented in [`TTT_RECEIPTS_INDEX.md`](./TTT_RECEIPTS_INDEX.md):

| # | Receipt | Result |
|---|---|---|
| 1 | 5-step proof-of-life | mechanism works |
| 2 | 30-step long-horizon stability | 28/30 applied, 0 OOM, parse 1.000 throughout |
| 3 | 50-step long-horizon stability (16-row probe) | 48/50 applied, steady-state from step 10 |
| 4 | Class-targeted v1 (full-CE loss) | regressed — diagnosed loss-formulation issue |
| 5 | Class-targeted v2 (action-weighted CE) — **skip class** | **+37.5 pp lift** in 16 steps |
| 6 | Class-targeted v2 (action-weighted CE) — **defer class** | **+75 pp lift** in 16 steps |
| 7 | Stratified TTT v2 (full 32-row holdout pre/post) | net +3.1 pp, no catastrophic regression |

**The architectural claim is empirically validated end-to-end on a real
LFM2.5-VL checkpoint.** Two-class average lift: +56.25 pp per pass.

**MuZero Liquid Track BC (10-seed sweep with LFM2.5-VL-450M as encoder):**
best `val_acc` 0.898 ± 0.049, range [0.825, 0.975]. See
[`MUZERO_LFM_EVAL.md`](./MUZERO_LFM_EVAL.md) and
[`MUZERO_SEED_SWEEP.md`](./MUZERO_SEED_SWEEP.md).

**Test-suite + quickstart:** `python scripts/quickstart.py` exits 0 on
Linux CI with all 209 tests passing. See `.github/workflows/tests.yml`.

### 4. Demo & Communication — 20%

- **Pitch video:** attached separately to the submission form (same video
  for both tracks; the LFM evidence appears on screen during shots 4-7).
  Source: [`VIDEO_SCRIPT.md`](./VIDEO_SCRIPT.md), built from
  [`fig/architecture_diagram.png`](./fig/architecture_diagram.png) and
  [`video_assets/`](./video_assets/).
- **Pinned operator-reviewed cases** (one per scenario pack):
  [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md). The Rotterdam case
  (urban_coastal_ambiguity, 48.78% cloud) is the architectural calibration
  loop made concrete: scaffold `accept` → trust `refine` → operator
  `accept`.
- **Live FastAPI server + dashboard** for an end-to-end interactive
  walkthrough. 7-step **Suggested Live Demo Flow** in
  [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md).

---

## Where to start (Liquid Track judging order)

1. [`SUBMISSION_TLDR.md`](./SUBMISSION_TLDR.md) — 1 page, 30 seconds
2. [`SUBMISSION_BRIEF.md`](./SUBMISSION_BRIEF.md) — full rubric mapping (this file is the Liquid Track summary)
3. [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md) — pinned operator cases
4. [`LFM_FINETUNE_METHODOLOGY.md`](./LFM_FINETUNE_METHODOLOGY.md) — recipe + holdout
5. [`LFM_TTT_POC.md`](./LFM_TTT_POC.md) — TTT receipts
6. [`TTT_RECEIPTS_INDEX.md`](./TTT_RECEIPTS_INDEX.md) — all 7 receipts in one nav page
7. [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md) — long-form architectural thesis

## Award package mapping (what the prize hardware unlocks)

The submission's wired-but-not-benchmarked Stage 3 lane targets the
**5 GPU hours on Orin 16 GB · 5 MB uplink · 10 MB downlink · 1 GB
storage for 1 month · 7 days ground-compute testing** allocated to the
prize. Specifically: long-horizon (100+ cycle) two-scope TTT under live
encounter outcomes, with the confirmed-outcome signal source from orbit
retrospective. The 7 published TTT receipts demonstrate the loop is
ready for that lane.

---

**HumanAI Convention** · [humanaiconvention.com](https://humanaiconvention.com)
On-orbit inference, continually refined by operator-labelled JSON within
uplink bandwidth parameters. Apache-2.0 weights · AGPL-3 code · public
training kernels.
