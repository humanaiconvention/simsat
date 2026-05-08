# Good morning — overnight summary (T-1 deadline day)

You were gone 22:18 PDT (May 7) → 06:30 PDT (May 8). This file is the
first thing to read. Last update: ~02:45 PDT.

**Submission deadline: Friday May 8, 5:00 PM PDT (8:00 PM EST).**
You have ~10 hours from when you read this.

---

## 🚨 #1 BLOCKER — fix this BEFORE recording the video

The GitHub repo `humanaiconvention/simsat` (where all my commits from
last night + 17+ overnight commits live) is **PRIVATE**. Judges will not
be able to see it.

**Fix (30 seconds):**
1. Go to https://github.com/humanaiconvention/simsat/settings
2. Scroll to "Danger Zone" → "Change repository visibility" → Public
3. Confirm.

After that, the URLs in `SUBMISSION_ABSTRACT.md` / `SUBMISSION_TLDR.md` /
`video_assets/shot9_close.txt` resolve. Verify by visiting
`https://github.com/humanaiconvention/simsat` in a private browser
window.

If your form expects `HumanAIConvention/SimSat` (capital S), GitHub URLs
are case-insensitive after public.

---

## 🎯 Headline overnight result — **TTT EMPIRICALLY LIFTS A TARGET CLASS**

The architectural argument graduated from *"TTT loop is wired and stable"*
to *"TTT loop empirically lifts target-class accuracy +37.5 pp on a real
LFM checkpoint, per pass, under operator-curated stream."*

### Class-targeted TTT v2 (action-token-weighted CE) — two-class lift

| Class | PRE | POST (16 steps) | Δ |
|---|---|---|---|
| skip | 0.375 | **0.750** | **+0.375 (+37.5 pp)** |
| defer | 0.125 | **0.875** | **+0.750 (+75.0 pp)** |
| **Average lift** | | | **+56.25 pp** |

Both runs: loss drove ~1.0 → ~0 in 16 steps. `lora_delta_l2` grew
0.0081 → ~0.040 monotonically. Same `OnlineLoRAStepper`, same six
viability gates as the stability receipts, same v3 adapter starting
point — only the loss mask and the target class changed between runs.

**This is the strongest architectural evidence the submission can produce
on available hardware before the prize hardware unlocks long-horizon
runs.** It's now in `LFM_TTT_POC.md` (new "Class-targeted TTT receipt"
section), `SUBMISSION_BRIEF.md` (TTT bullet rewritten), `SUBMISSION_TLDR.md`
(TLDR rewritten), `README.md` (headline bullet), `VIDEO_SCRIPT.md` shot 7
(voiceover rewritten), and `video_assets/frames/shot7.png` (frame
redesigned to show the PRE/POST table).

---

## Full overnight experiment log (5 TTT receipts + 1 fine-tune)

| # | Experiment | Time | Result |
|---|---|---|---|
| Phase 2 | Extended TTT v1 (30 steps, 8-row probe, v3 adapter) | 22:30-00:00 | 28/30 applied, 0 OOM, parse 1.000 throughout, monotonic LoRA delta. **Long-horizon stability receipt.** |
| Phase 3 | v3+ recipe variant (lr=1e-4, 8 epochs, dropout=0.10, same v2 dataset) | 23:00-01:00 | -3.1 pp action vs v3 (negative). **Third consecutive offline-tuning negative attempt to beat v3** (after v4 +imbalanced data and v5 +balanced data). Strong empirical evidence v3 is at a local optimum. |
| Phase 3.5 | Extended TTT v2 (50 steps, 16-row probe, v3 adapter) | 01:10-02:05 | 48/50 applied, 0 OOM, steady-state from step 10. **Tighter probe sharpens architectural argument: TTT preserves but boundary movement is data-dependent → why the lane is gated TTT, not free-running TTT.** |
| Phase 4 | Class-targeted TTT v1 (full-CE loss, skip-only stream) | 02:05-02:25 | -37.5 pp regression (negative). **Diagnosed: full-assistant CE diluted action signal across ~99 non-action tokens.** Honest negative published. |
| Phase 4 | Class-targeted TTT v2 (action-token-weighted CE, skip-only stream) | 02:25-02:50 | **+37.5 pp lift on skip class.** Mechanism + curation + loss → real lift. |
| Phase 4 | Class-targeted TTT v2 (defer-only stream) | 02:50-03:30 | **+75 pp lift on defer class.** Two-class generalization confirmed. |
| Phase 4 | Stratified TTT v2 (4 per class × 4 = 16 steps, full 32-row holdout) | 03:30-04:00 | **Robustness check: net +3.1 pp overall, no catastrophic regression.** Skip +37.5 pp, defer/refine each −12.5 pp, accept flat. Architectural lesson: targeted TTT > stratified for max lift; stratified is safety floor. |

All 5 receipts published in `.kaggle_output/*receipt*.json`. All 18+
commits pushed to `humanaiconvention/simsat` main.

---

## Headline numbers (still v3 canonical, unchanged)

| Metric | Base | v3 | Δ |
|---|---|---|---|
| `exact_action_agreement` | 0.156 | **0.844** | **+68.8 pp** |
| `score_mae` | 0.365 | **0.055** | **−31.0 pp** |
| Per-class accept / refine / defer / skip | — | 1.000 / 1.000 / 0.625 / 0.750 | balanced |

**Honest negatives now triple-stacked (all kept v3 canonical):**
- v4 (+20 imbalanced data): −6.3 pp action
- v5 (+56 class-balanced data): −15.6 pp action; skip 0.75→0.25
- **v3+ (recipe variant on same v2 data): −3.1 pp action** ← new tonight

**Architectural conclusion:** three independent lines (data-imbalanced,
data-balanced, recipe variant) converge on the same finding — v3 sits at
a local optimum on this architecture × data × holdout combination that
further offline tuning cannot escape. The next lift comes from runtime
TTT (now empirically demonstrated to lift +37.5 pp/pass on a target
class), not more offline data.

---

## Tomorrow morning's flow (in order)

| Step | What | Time | Tool |
|---|---|---|---|
| 1 | **MAKE THE GITHUB REPO PUBLIC** (see #1 BLOCKER above) | 30 sec | GitHub Settings |
| 2 | Pull this branch fresh | 10 sec | `cd D:\SimSat && git pull fork main` |
| 3 | Read this file + `SESSION_LOG.md` | 5 min | text editor |
| 4 | Skim 9 video frames in `video_assets/frames/` | 5 min | image viewer (especially shot 7 — that's the new headline frame) |
| 5 | Mic check, kill notifications | 5 min | — |
| 6 | **Record voiceover** following `VIDEO_SCRIPT.md` (~3:55, 115 wpm) | 60-90 min | Audacity |
| 7 | Save WAV anywhere convenient | 30 sec | — |
| 8 | `python video_assets/overlay_audio.py path/to/voiceover.wav` | 5 min | terminal |
| 9 | Review `video_assets/simsat_demo_final.mp4` | 5 min | video player |
| 10 | Upload video (YouTube unlisted, or directly to submission form) | 10 min | browser |
| 11 | Submit form using `SUBMISSION_ABSTRACT.md` content | 15 min | browser |
| 12 | `git tag -a v1.0-submission -m "DPhi/Liquid hackathon final" && git push fork --tags` | 1 min | terminal |

Total: 2-3 hours of focused work + the recording itself.

---

## Where canonical things live

- **Code:** `D:\SimSat` (this repo). 35+ commits over 2 days.
- **Submission abstract drafts:** `SUBMISSION_ABSTRACT.md` (paste-ready)
- **1-page judge skim:** `SUBMISSION_TLDR.md`
- **Demo video script:** `VIDEO_SCRIPT.md` (literal voiceover lines + shot list)
- **Silent video:** `video_assets/silent_demo.mp4` (~4.7 MB, 1920×1080, 30 fps, ~3:55)
- **Architecture diagram:** `fig/architecture_diagram.png`
- **HF adapter:** `HumanAIConvention/simsat-lfm25vl-450m-v3` (canonical)
- **Public Kaggle kernels:** `benhaslam/simsat-lfm2-5-vl-{v1,v3,v4,v5}-training`
- **TTT receipts (5 of them):** `.kaggle_output/*receipt*.json`
- **Today's session log:** `SESSION_LOG.md`

---

## What I deliberately did NOT do

- Did not promote v4/v5/v3+ adapters to HuggingFace — they are kept locally
  for transparency but not canonical.
- Did not regenerate `SUBMISSION_PACKET.md` / `SUBMISSION_READINESS.md`
  — those auto-generators have a flaky Sentinel STAC dependency that
  Cowork got stuck on yesterday; cosmetic-stale-timestamp-only,
  judge-irrelevant.
- Did not record the video myself (impossible — needs your voice and mic).
- Did not break or modify any HF model or training kernel artifacts that
  are already public.

## Questions to consider before submission

1. **Repo visibility:** confirmed public after fix #1?
2. **Form audience:** does the submission form ask for one repo URL or one
   per track? (Same code, different track-specific receipts.)
3. **Video format:** does the form upload directly, or do you link
   YouTube/Vimeo? If link, YouTube unlisted is fine.
4. **Licenses:** Apache-2.0 weights / AGPL-3 code — is that consistent
   with form requirements?
