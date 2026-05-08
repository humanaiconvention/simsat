# SimSat — General AI Track Entry

**Audience:** judges scoring the General AI Track of the AI in Space
Hackathon (DPhi Space × Liquid AI). Rubric: 20 / 25 / 35 / 20 (Use of
Satellite Imagery / Innovation & Problem-Solution Fit / Technical
Implementation / Demo).

**Same repo, same code as the Liquid Track entry** — see
[`LIQUID_TRACK.md`](./LIQUID_TRACK.md). The architectural contribution is
shared; what differs is which model backend fills the encoder seat. For
this track, the canonical backend is **Gemma-4-E2B**, and the broader
contribution is the **model-agnostic ObservationVLA scaffold** that any
VLM emitting the eight-key JSON contract plugs into.

---

## What this submission delivers for the General AI Track

### 1. Use of Satellite Imagery — 20%

*(multispectral and space-acquisition constraints explicitly rewarded
in this rubric line)*

- **Sentinel-2 is the core data source** across all scenario packs —
  same coverage as the Liquid Track entry.
- **Multiple spectral bands used in operations** (rubric: "rewarded"):
  - **NDVI:** B04, B08
  - **SWIR ratio:** B11, B12 — soil moisture / organic-matter proxy
  - **EVI:** B08A, B05 — canopy structure
  Most submissions use RGB only; SimSat's `pedospheric_integrity`
  scenario pack is *built on* the spectral-biochemical register.
- **Temporal continuity / large volumes / limited downlink** (rubric:
  "preference is given to approaches that reflect..."): the
  encounter-window architecture *is* the continuous-stream model —
  windows arrive on minutes-cadence; selective downlink (only after all
  six viability gates pass + high-confidence accept) is built into the
  architecture, not bolted on.
- **Two observational registers** (geometric/structural +
  spectral-biochemical) operating identically through the same pipeline
  — demonstrates cross-domain generality on Sentinel data.

**Eval evidence:**
- 152 operator-reviewed Sentinel-2 tiles across all 4 scenario packs
  ([`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md))
- 248-window encounter eval on real Sentinel-backed geometry
  ([`ENCOUNTER_EVAL.md`](./ENCOUNTER_EVAL.md))

### 2. Innovation & Problem-Solution Fit — 25%

*(must clearly justify why on-orbit)*

**Three stacked physical constraints, each independently sufficient to
force the loop on-board:**

1. No ground-truth validator under distribution shift (sensors drift,
   labels lag).
2. 5 MB uplink budget cannot ferry weight updates between encounters.
3. Next encounter window arrives in minutes — no round-trip budget for
   ground retraining.

**What on-board compute uniquely enables:** stacked TTT under six
non-compensatory viability gates. The viability gates are the substitute
for human-in-the-loop on a satellite — every accepted update carries
quality-verified, magnitude-bounded, bias-checked information about the
world before touching model state. None of that is possible from the
ground at the required cadence.

**Distribution-shift demo** quantifies the on-orbit improvement
mechanism: 94% of polar-regime improvement in **one cycle** of operator
feedback after a coastal → polar shift
([`ttt_stability_analysis.md`](./ttt_stability_analysis.md)).

**Cross-track architectural validation:** the same TTT loop and
viability gates work for the LFM2.5-VL backend (Liquid Track) — see
[`TTT_RECEIPTS_INDEX.md`](./TTT_RECEIPTS_INDEX.md). The 7-receipt
development arc on the LFM backend (including +37.5 pp / +75 pp
class-targeted lifts on a real VLM checkpoint) is also evidence for the
General AI Track's "this architecture is real, not vapor" claim.

### 3. Technical Implementation — 35%

*(fine-tuning is "(optional)"; conceptual innovation + system design
co-equal)*

**App runs cleanly:** `python scripts/quickstart.py` exits 0 on Linux CI
(see `.github/workflows/tests.yml`); 209 unit tests pass; no-Mapbox-safe.

**Conceptual innovation:** stacked TTT + six-gate viability filter +
two-register architecture; not "adapt an existing model."

**System design — model-agnostic backend factory:** the
`OBSERVATION_VLA_BACKEND` env var swaps backends without code changes.
Five backends shipped:
- `clip_local` (CLIP ViT-B/32 baseline)
- `gemma4` (Gemma-4-E2B SimSat fine-tune — canonical for this track)
- `transformers_vlm` (HF-generic VLM adapter)
- `genesis` (collaborator integration)
- `tesseract_t3` (collaborator integration)
- `tesseract_t3_heatmap` and `heatmap` (auxiliary)

The 8-key JSON contract is documented in
[`COLLABORATOR_GUIDE.md`](./COLLABORATOR_GUIDE.md). Two independent
collaborators (Genesis, Tesseract T3) integrated against this contract
during the build window.

**Optional fine-tune (delivered) — Gemma-4-E2B v11:**

| Eval frame | Result |
|---|---|
| **In-distribution (N=37, accept↔refine geometric)** | exact **0.86** (32/37) · useful **0.97** · MAE **0.13** |
| Per-pack (in-distribution) | disaster=1.00, maritime=0.92, urban-coastal=0.71 |
| **Cross-distribution (N=152, balanced 4-class)** | exact **0.30** — distribution-shift evidence motivating the on-orbit TTT lane |

- **Adapter on Hugging Face:** [`HumanAIConvention/simsat-gemma4-v11`](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11)
- **Public training kernel:** `benhaslam/simsat-gemma4-v1-training` on Kaggle
- **Adapter integrity:** 410/410 dynamic LoRA tensor sanity gate **PASS**
  ([`V11_AUDIT.md`](./V11_AUDIT.md))
- **v12 retrain on dataset v4 was attempted and regressed** (35% parse
  rate vs v11's ~100% — see [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) #28).
  Honest negative published; v11 retained as canonical.

**The 56-point gap between v11 in-distribution (0.86) and
cross-distribution (0.30) is the architectural argument made concrete:**
static fine-tuning sets where drift starts, but only on-orbit TTT under
viability gates can adapt to drift in flight. v12 attempted to close
this gap with offline retraining and produced a parse-rate regression
instead, confirming the runtime-adaptation thesis.

**Inference benchmarks** (`benchmark_results/BENCHMARK_RESULTS.md`):
Tesla T4, float16, N=100 timed passes, 3 warmup. Mean latency
**9,455.7 ms**, p95 **9,620.8 ms**, std **90.2 ms**, **12.8 tokens/sec**.

### 4. Demo & Communication — 20%

Same demo assets as the Liquid Track:

- **Pitch video** (attached to submission form). Source:
  [`VIDEO_SCRIPT.md`](./VIDEO_SCRIPT.md), built from
  [`fig/architecture_diagram.png`](./fig/architecture_diagram.png) and
  [`video_assets/`](./video_assets/).
- **Pinned operator-reviewed cases** (one per scenario pack):
  [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md). The Rotterdam
  case demonstrates the scaffold ↔ trust ↔ operator three-way
  calibration loop end-to-end at 48.78% cloud cover.
- **Live FastAPI server + dashboard** for an end-to-end interactive
  walkthrough.
- 7-step **Suggested Live Demo Flow** in
  [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md).

---

## Where to start (General AI Track judging order)

1. [`SUBMISSION_TLDR.md`](./SUBMISSION_TLDR.md) — 1 page, 30 seconds
2. [`SUBMISSION_BRIEF.md`](./SUBMISSION_BRIEF.md) — full rubric mapping (this file is the General AI Track summary)
3. [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md) — pinned operator cases
4. [`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md) — Gemma-4 v11 eval (in-distribution + cross-distribution)
5. [`COLLABORATOR_GUIDE.md`](./COLLABORATOR_GUIDE.md) — model-agnostic backend integration
6. [`V11_AUDIT.md`](./V11_AUDIT.md) — adapter sanity gate audit
7. [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md) — long-form architectural thesis
8. *Bonus from the Liquid Track lane that also strengthens this track:*
   [`LFM_TTT_POC.md`](./LFM_TTT_POC.md) — TTT empirically lifts target-class
   accuracy on a real VLM checkpoint, +37.5 pp / +75 pp on two classes.

## Why submit the same repo to both tracks

The submission's central architectural claim — *governed continual
learning under viability gates* — is **model-agnostic** by design. The
same scaffold, the same six gates, the same TTT loop, and the same
operator-reviewed eval pipeline work whether you plug in Gemma-4-E2B
(General AI Track canonical), LFM2.5-VL-450M (Liquid Track canonical),
CLIP, or any future VLM emitting the eight-key contract. Splitting the
repo would force two copies of the same code; keeping it unified shows
the contribution.
