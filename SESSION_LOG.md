# Session Log — 2026-05-07 (T-1 to submission)

Operator: ben. Submission deadline: Friday May 8, 5:00 PM PDT (8:00 PM EST)
per the official hackathon banner. Session started ~12:00 PDT, autonomous
overnight phase begins ~22:30 PDT.

This file is a chronological summary of what shipped today, in commit
order. The git history is the canonical record; this file is the narrative.

---

## Submission deliverables landed (v3 canonical)

### LFM2.5-VL-450M LoRA fine-tune track

**4 trained adapters, 1 promoted as canonical:**

| Adapter | Train rows | Holdout exact_action | score_mae | Status | On HF? |
|---|---|---|---|---|---|
| **v3** | 165 | **0.844** | **0.055** | **canonical** | ✓ [link](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3) |
| v1 (Run 14) | 109 | 0.656 | 0.102 | superseded | ✓ [link](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v1) (marked superseded) |
| v1 + rep_penalty (Run A) | 109 | 0.750 | 0.080 | proven decode-time only | (same v1 weights) |
| v4 | 185 | 0.781 | 0.066 | negative result | local on Kaggle, not promoted |
| v5 | 241 | 0.688 | 0.084 | negative result | local on BEAST, not promoted |

**Two consecutive +data negatives (v4 imbalanced, v5 class-balanced) confirm
v3 is at the inflection point of the training-data curve for this
architecture / holdout combination.** That's the architectural argument
for runtime TTT over more offline data.

### TTT proof-of-life receipt

Closed the prior `LFM_TTT_POC.md` "wired, requires live stream" gap with a
real-model run (BEAST RTX 2080):
- 5/5 attempted online-LoRA gradient steps applied
- 0 viability gates triggered
- post-MAE held perfect on stratified probe (1.000 / 0.000 / 1.000)
- lora_delta_l2 grew monotonically 0.0008 → 0.0021
- Step 6 hit clean CUDA-OOM at 8 GB ceiling (caught by try/except)

Receipt: [`.kaggle_output/ttt_proof_of_life_receipt.json`](.kaggle_output/ttt_proof_of_life_receipt.json).

---

## Today's commit timeline

| Commit | Title (one-line) |
|---|---|
| `2218eb2` | LFM v3: 51 new operator reviews + decode hardening, kernel running |
| `10e8fd8` | v3 canonical adapter shipped: action 0.844, MAE 0.055, refine + accept perfect |
| `81eb29d` | TTT proof-of-life receipt + v4 kernel scaffold (training in flight) |
| `f9a083b` | v4 trained: -6.3pp action vs v3 (negative result, kept v3 canonical) |
| `76ab508` | v2 prep hardening: drop empty/corrupt PNGs before including rows |
| `d0f6563` | v5 dataset (241 train, defer 34→51) + kernel + local BEAST trainer |
| `8a97e21` | docs: SUBMISSION_BRIEF / CHALLENGE_ENTRY / README updated to cite v3 as canonical |
| `9faa512` | chore: archive root-level clutter, tighten .gitignore |
| `e542df8` | chore: archive orphaned notebooks |
| `648035e` | docs: full-repo staleness sweep (4 docs, "in flight" claims fixed) |
| `4621f7e` | ci: remove hardcoded D:/SimSat paths from local_train_v5.py docstring |
| `32b2594` | chore: archive review_assets/ (orphan) |
| `57a3c2f` | chore(pre-commit): exclude .pre-commit-config.yaml from hardcoded-paths hook |
| `2ee1ab6` | docs: submission scaffolding — abstract + architecture diagram + video script |
| `8efe94a` | video: pre-stage architecture PNG + 5 shot panels |
| `6222465` | video: pre-built silent demo + voiceover overlay tooling |
| `1a1edbc` | docs: rubric-driven improvements (banner, TLDR, pytest.ini fix) |
| `353d4e8` | v5 trained: -15.6pp action vs v3 (skip cratered) |

18 commits on `humanaiconvention/simsat` main.

---

## Operator review sessions completed today

Three review sessions via `gallery_review.html`:

| Session | Time | New traces | Labels | Notes |
|---|---|---|---|---|
| 1 | morning | 51 (8 leftover + 50 from `materialize_adversarial.py`) | 16 a / 31 r / 4 d / 0 s | Planner-marginal candidates, no skip overrides |
| 2 | early afternoon | 20 from defer-targeted re-plan | 5 a / 15 r / 0 d / 0 s | 50-80% cloud band, all confirmed refine |
| 3 | early evening | 56 from `materialize_defer_targeted.py` | 8 a / 30 r / 17 d / 1 s | Defer envelope (cloud 55-90% + bad geometry) |

Total: 127 new operator reviews. Pool grew 145 → 272 reviewed.

---

## Documents shipped or updated

| Doc | Change |
|---|---|
| `README.md` | Banner added, v3 numbers in headline list, two-negatives framing on v4+v5 |
| `SUBMISSION_BRIEF.md` | Lines 192-198 rewritten for v3 canonical + v4/v5 negatives + TTT receipt |
| `SUBMISSION_TLDR.md` | NEW — 1-page judge skim |
| `SUBMISSION_ABSTRACT.md` | NEW — form-fillable elevator + 3-paragraph + per-track mapping |
| `LFM_FINETUNE_METHODOLOGY.md` | New §6.4 (v1 numbers), §6.5 (v3 canonical), §6.6 (v4 negative), §6.7 (v5 negative) |
| `LFM_TTT_POC.md` | New "Live receipt — 2026-05-07" section closing the prior gap |
| `CHALLENGE_ENTRY.md` | Lines 69, 92, 93 updated to reflect trained encoder LoRA (was "not trained") |
| `OBSERVATION_VLA_EVAL.md` | Lines 65, 83 stale "in progress" claims fixed |
| `KNOWN_ISSUES.md` | Item 29 marked Resolved with full v1→v5 timeline |
| `VIDEO_SCRIPT.md` | NEW — 9-shot script with literal voiceover lines |
| `fig/ARCHITECTURE.md` | NEW — Mermaid + ASCII source for the architecture diagram |
| `fig/architecture_diagram.png` | NEW — rendered PNG for the demo video |
| `video_assets/render_frames.py` | NEW — produces 9 styled 1920×1080 PNG frames |
| `video_assets/build_video.py` | NEW — stitches frames into silent_demo.mp4 |
| `video_assets/overlay_audio.py` | NEW — drops user voiceover onto silent video |
| `video_assets/silent_demo.mp4` | NEW (gitignored) — 4.5 MB, 1920×1080, 30 fps, 3:50 |

---

## Code changes

| File | Change |
|---|---|
| `src/sim/observation_vla/lfm_ttt.py` | (already in-tree before today) — exercised end-to-end via TTT proof-of-life run |
| `notebooks/kaggle-simsat-lfm-v1/local_eval_runA.py` | NEW — Run A re-eval on v1 with `repetition_penalty=1.05` |
| `notebooks/kaggle-simsat-lfm-v1/ttt_proof_of_life.py` | NEW — VLA-layer TTT receipt script |
| `notebooks/kaggle-simsat-lfm-v3/` | NEW — v3 training kernel (Kaggle T4, public) |
| `notebooks/kaggle-simsat-lfm-v4/` | NEW — v4 training kernel (Kaggle T4, public, negative result) |
| `notebooks/kaggle-simsat-lfm-v5/` | NEW — v5 training kernel + `local_train_v5.py` (BEAST overnight, negative result) |
| `datasets/simsat-lfm-v2/` | NEW — 165 train rows (v3 used this) |
| `datasets/simsat-lfm-v3/` | NEW — 185 train rows (v4 used this) |
| `datasets/simsat-lfm-v5/` | NEW — 241 train rows (v5 used this) |
| `scripts/batch_materialize.py` | NEW — fans out /encounter/decision/{id}/{materialize,assess} |
| `scripts/materialize_adversarial.py` | NEW — multi-horizon planner sweep for marginal-cloud candidates |
| `scripts/materialize_defer_targeted.py` | NEW — narrow defer-envelope candidate generator |
| `scripts/gen_gallery_thumbs.py` | NEW — fetches Sentinel tiles for review-tool thumbnails |
| `pytest.ini` | Removed hardcoded `D:/SimSat/.test_tmp` basetemp + `tmp_path_retention_policy=all` |
| `.gitignore` | Added `.kaggle_output*/`, `/labels*.json`, `/gallery_review.html`, `/test_sentinel*.png`, `scripts/_*_snapshot.json`, `video_assets/*.mp4`, `video_assets/frames/*.png` |
| `.pre-commit-config.yaml` | Excluded self-reference of the hardcoded-paths regex |

---

## What's running overnight (Phase 2-4)

Set up to run autonomously 22:30 PDT → 06:30 PDT:

- **Phase 2 (~3 hr):** Extended TTT run on v3 adapter, target 30+ steps (long-horizon
  stability). Closes the explicit "no demonstration of long-horizon stability"
  gap in `LFM_TTT_POC.md`.
- **Phase 3 (~3 hr):** Recipe variant if Phase 2 succeeded — try a v3 retrain
  with different hyperparameters to map the recipe space.
- **Phase 4 (~1 hr):** Final consolidation, `WAKEUP_SUMMARY.md` write, final
  commit + push.

Commits will be pushed throughout. Look at `WAKEUP_SUMMARY.md` first when
you're back at 06:30.

---

## Critical action item for tomorrow morning (BEFORE submission)

🚨 **The GitHub repo `humanaiconvention/simsat` is currently PRIVATE.**

All my commits today went to that private repo. The submission docs reference
`github.com/HumanAIConvention/SimSat` (which 404s) and the archived
`humanaiconvention/SimSat-1` (read-only). **Judges cannot access the repo as
it stands.**

**Three options to fix tomorrow:**

1. **Best:** Make `humanaiconvention/simsat` PUBLIC in GitHub Settings.
   This is one click and immediately makes all 18 commits visible.
2. **Alternative:** Unarchive `SimSat-1` and force-push there.
3. **Fallback:** Create a fresh public repo `HumanAIConvention/SimSat` and
   push there.

Option 1 is fastest (~30 sec). Recommend doing that first thing tomorrow.
After that's done, all `github.com/HumanAIConvention/SimSat` references in
SUBMISSION_ABSTRACT.md / SUBMISSION_TLDR.md / video shot 9 / etc. need to
be updated to point at `github.com/humanaiconvention/simsat` (note casing).
Or change the GitHub display name to match.
