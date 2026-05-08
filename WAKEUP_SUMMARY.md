# Good morning — overnight summary (T-1 deadline day)

You were gone 22:18 PDT → 06:30 PDT. This file is the first thing to read.

**Submission deadline: Friday May 8, 5:00 PM PDT (8:00 PM EST).**
You have ~10 hours from when you read this.

---

## 🚨 #1 BLOCKER — fix this BEFORE recording the video

The GitHub repo `humanaiconvention/simsat` (where all 18+ commits from
yesterday + my overnight commits live) is **PRIVATE**. Judges will not be
able to see it.

**Fix (30 seconds):**
1. Go to https://github.com/humanaiconvention/simsat/settings
2. Scroll to "Danger Zone" → "Change repository visibility" → Public
3. Confirm.

After that's done, all the URLs in `SUBMISSION_ABSTRACT.md` /
`SUBMISSION_TLDR.md` / `video_assets/shot9_close.txt` should resolve. Verify
by visiting `https://github.com/humanaiconvention/simsat` in a private
browser window.

If the case in your form expects `HumanAIConvention/SimSat` (capital S),
you can either rename the repo or just paste the lowercase URL — GitHub
URLs are case-insensitive after public.

---

## Overnight: what ran

(Filled in at end of the run; auto-updated.)

### Phase 2 — Extended TTT on v3 adapter

**Goal:** close the "no demonstration of long-horizon stability" gap in
`LFM_TTT_POC.md`. The trust-layer has 100-cycle stability evidence; the
VLA-layer only had a 5-step receipt.

**Result:** *(to be filled in)*

Receipt: `.kaggle_output/extended_ttt_receipt.json`.

### Phase 3 — Recipe variant *(if Phase 2 succeeded)*

*(to be filled in)*

### Phase 4 — Final consolidation

*(to be filled in)*

---

## Tomorrow morning's flow (in order)

| Step | What | Time | Tool |
|---|---|---|---|
| 1 | **MAKE THE GITHUB REPO PUBLIC** (see above) | 30 sec | GitHub Settings |
| 2 | Pull this branch fresh | 10 sec | `cd D:\SimSat && git pull fork main` |
| 3 | Read this file + `SESSION_LOG.md` | 5 min | text editor |
| 4 | Skim 9 video frames in `video_assets/frames/` to verify they look right | 5 min | image viewer |
| 5 | Mic check, kill notifications | 5 min | — |
| 6 | **Record voiceover** following `VIDEO_SCRIPT.md` (~3:50, 115 wpm) | 60-90 min | Audacity |
| 7 | Save WAV anywhere convenient | 30 sec | — |
| 8 | `python video_assets/overlay_audio.py path/to/voiceover.wav` | 5 min | terminal |
| 9 | Review `video_assets/simsat_demo_final.mp4` | 5 min | video player |
| 10 | Upload video (YouTube unlisted, or directly to submission form) | 10 min | browser |
| 11 | Submit form using `SUBMISSION_ABSTRACT.md` content | 15 min | browser |
| 12 | `git tag -a v1.0-submission -m "DPhi/Liquid hackathon final" && git push fork --tags` | 1 min | terminal |

Total: 2-3 hours of focused work + the recording itself.

---

## Where canonical things live

- **Code:** `D:\SimSat` (this repo)
- **Submission abstract drafts:** `SUBMISSION_ABSTRACT.md` (paste-ready)
- **Demo video script:** `VIDEO_SCRIPT.md` (literal voiceover lines + shot list)
- **Silent video to overlay:** `video_assets/silent_demo.mp4` (4.5 MB, 1920×1080, 30 fps, 3:50)
- **Architecture diagram:** `fig/architecture_diagram.png`
- **HF adapter:** `HumanAIConvention/simsat-lfm25vl-450m-v3` (canonical)
- **Public Kaggle kernels:** `benhaslam/simsat-lfm2-5-vl-{v1,v3,v4,v5}-training`

## Headline numbers to remember

| Metric | Base | v3 | Δ |
|---|---|---|---|
| `exact_action_agreement` | 0.156 | **0.844** | **+68.8 pp** |
| `score_mae` | 0.365 | **0.055** | **−31.0 pp** |
| Per-class accept / refine / defer / skip | — | 1.000 / 1.000 / 0.625 / 0.750 | balanced |

**Honest negatives (both kept v3 canonical):**
- v4 (+20 imbalanced rows): −6.3 pp action
- v5 (+56 class-balanced rows): −15.6 pp action; skip 0.75→0.25 cratered

**Architectural conclusion:** two consecutive +data negatives confirm v3 is
at the inflection point on this architecture / holdout combination. The
runtime-TTT lane is empirically motivated, not just claimed.

---

*Filled in by Claude during the overnight session, last write at:*
*— `<wakeup_timestamp>` —*
