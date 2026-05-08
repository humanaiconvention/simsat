# SimSat Demo Video — Recording Script

**Target:** 3 min 50 sec. Single-take or two-take. 1080p MP4. Voiceover over
screen recording + still frames. Submit alongside the abstract.

**Tone:** confident, technical, honest about what's published vs what's the
prize-hardware lane. No hype, no "revolutionary" — let the +68.8 pp action
lift speak for itself.

---

## Shot list (one row per visual change on screen)

| # | Time | What's on screen | What's said (voiceover) | Voiceover length |
|---|---|---|---|---|
| 1 | 0:00-0:30 | **Black background. Phi-with-dot mark fades in white over 2s (0:00-0:02), holds. At ~0:04-0:05 the "Human AI / Convention" wordmark crossfades in below the mark — timed to land as the founder says "HumanAI Convention". Full lockup holds to 0:30.** | "I'm Ben Haslam, founder of the HumanAI Convention — a humble project to enable mass human flourishing through robust, ethical training data. The HumanAI Convention's hypothesis is that the synthetic data AI scales on must be balanced with data from human lived experience, and an ethical framework — humans living their best lives — provides the optimal data. That's the **Viability Convention** as I hypothesize it." | 30 s |
| 2 | 0:30-0:58 | Cross-fade to SimSat title card with Rotterdam Sentinel band | "This proposal takes that hypothesis and operationalizes it through SimSat — a working prototype where the Convention's six non-compensatory viability gates govern continual learning on a real-world problem. We chose satellite encounter triage because the constraints are physical: minutes between decisions, no round-trip to ground, an uplink too small to ferry weight updates. *What it can ferry is JSON.* The architecture is the contribution; the satellite is the test case." | 28 s |
| 3 | 0:20-0:50 | Architecture diagram top half from `fig/ARCHITECTURE.md` (Sentinel tile → LFM2.5-VL-450M encoder → trust layer + scaffold → action) | "One vision-language tile encoder — Liquid AI's LFM2.5-VL-450M — feeds a planner over encounter windows. A trust layer scores each candidate against geometry, priority, cloud, and visibility. Six viability gates govern any continual learning that runs on top." | 28 s |
| 4 | 0:50-1:20 | Cut to a terminal showing the v3 holdout report. Highlight rows with on-screen overlays | "Headline: we fine-tuned the LFM encoder with LoRA on 165 operator-reviewed Sentinel-2 tiles. Base model exact action agreement on a balanced 32-row holdout: 0.156. Tuned: zero point eight four four. Plus sixty-eight point eight points." | 28 s |
| 5 | 1:20-1:35 | Same screen — focus on score MAE row | "Score MAE drops from 0.365 to 0.055 — about a thirty-one-point reduction in band-mapped prediction error. Per-class accept and refine are perfect. The adapter is on Hugging Face under Apache-2.0." | 14 s |
| 6 | 1:35-2:30 | Cut to `submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png` full-frame. Side panel showing the Rotterdam case row from `SUBMISSION_CASEBOOK.md` (scaffold accept → trust refine → operator accept) | "Rotterdam, forty-eight point seven eight percent cloud cover. The scaffold says accept — geometry is good. The trust layer disagrees, flips it to refine — the cloud might hide containers. The operator confirms accept: the visible fifty-one percent of the basin was operationally enough. This is the calibration loop. The same loop runs on every window." | 55 s |
| 7 | 2:30-3:20 | Cut to terminal: class-targeted TTT v2 two-class result panel | "VLA-layer test-time training, validated end-to-end. Stability first: fifty steps on the v three adapter, forty-eight applied, zero divergence, parse rate one-point-zero throughout. Then the headline result, on two classes. Sixteen skip-class encounters streamed: skip lifted from zero-point-three-seven-five to zero-point-seven-five — plus thirty-seven points. Sixteen defer-class encounters streamed: defer lifted from zero-point-one-two-five to zero-point-eight-seven-five — plus seventy-five points. Average lift across two classes, plus fifty-six points in sixteen steps. TTT under operator-curated stream empirically lifts target-class accuracy per pass on TWO independent classes — the architectural claim, validated on a real LFM checkpoint." | 50 s |
| 8 | 3:05-3:30 | Cut to the v3 / v4 / v5 / v3+ comparison panel | "We tried three different ways to beat v three. More imbalanced data: minus six point three. More class-balanced data: minus fifteen point six. A different recipe — lower learning rate, longer schedule, more dropout — minus three point one. Three independent angles, same conclusion: v three sits at a local optimum on this architecture and this holdout. That's empirical evidence for the runtime-TTT lane, not just an architectural claim. All three negatives are published in the methodology doc." | 25 s |
| 9 | 3:30-3:55 | Final card: repo URL + HF model URLs stacked. White on black | "On-orbit inference, continually refined by operator-labelled JSON within uplink bandwidth parameters. Apache-2.0 weights on Hugging Face. Public training kernels on Kaggle. AGPL-3 code on GitHub. Built for the prize hardware — Orin sixteen gig is what unlocks the long-horizon adaptation we've already demonstrated end-to-end. Thanks." | 22 s |

**Total: 4 min 43 sec** (silent demo: 30 + 28 + 30 + 30 + 15 + 55 + 45 + 25
+ 25 = 283s). The HumanAI Convention logo cold-open (shot 1) plus the
"viability convention → SimSat as test case" bridge (shot 2) frame the
numbers as evidence; the close card (shot 9) leads with humanaiconvention.com.

---

## Full voiceover script (run-through, no shot breaks)

> A satellite has minutes, not hours, between encounter windows. There's
> no round-trip to ground for retraining. The uplink budget can't ferry
> weight updates either. *What it can ferry is JSON.* SimSat is
> on-orbit inference, continually refined by operator-labelled JSON
> within uplink bandwidth parameters.
>
> One vision-language tile encoder — Liquid AI's LFM-two-point-five-VL-450M —
> feeds a planner over encounter windows. A trust layer scores each
> candidate against geometry, priority, cloud, and visibility. Six viability
> gates govern any continual learning that runs on top.
>
> Headline: we fine-tuned the LFM encoder with LoRA on a hundred
> sixty-five operator-reviewed Sentinel-2 tiles. Base model exact action
> agreement on a balanced thirty-two-row holdout: zero point one
> five six. Tuned: zero point eight four four. Plus sixty-eight point
> eight points.
>
> Score MAE drops from zero-point-three-six-five to zero-point-zero-five-five
> — about a thirty-one-point reduction in band-mapped prediction error.
> Per-class accept and refine are perfect. The adapter is on Hugging Face
> under Apache-2.0.
>
> Rotterdam. Forty-eight point seven eight percent cloud cover. The scaffold
> says accept — geometry is good. The trust layer disagrees, flips it to
> refine — the cloud might hide containers. The operator confirms accept:
> the visible fifty-one percent of the basin was operationally enough. This
> is the calibration loop. The same loop runs on every window.
>
> VLA-layer test-time training, validated end-to-end. Stability first:
> fifty steps on the v three adapter, forty-eight applied, zero
> divergence, parse rate one-point-zero throughout. Then the headline
> result, on two classes. Sixteen skip-class encounters streamed: skip
> lifted from zero-point-three-seven-five to zero-point-seven-five —
> plus thirty-seven points. Sixteen defer-class encounters streamed:
> defer lifted from zero-point-one-two-five to zero-point-eight-seven-
> five — plus seventy-five points. Average lift across two classes,
> plus fifty-six points in sixteen steps. TTT under operator-curated
> stream empirically lifts target-class accuracy per pass on TWO
> independent classes — the architectural claim, validated on a real
> LFM checkpoint.
>
> We tried three different ways to beat v three. More imbalanced data:
> minus six point three. More class-balanced data: minus fifteen point
> six. A different recipe — lower learning rate, longer schedule, more
> dropout — minus three point one. Three independent angles, same
> conclusion: v three sits at a local optimum on this architecture and
> this holdout. That's empirical evidence for the runtime-TTT lane, not
> just an architectural claim. All three negatives are published in the
> methodology doc.
>
> On-orbit inference, continually refined by operator-labelled JSON
> within uplink bandwidth parameters. Apache-2.0 weights on Hugging
> Face. Public training kernels on Kaggle. AGPL-3 code on GitHub. Built
> for the prize hardware — Orin sixteen gig is what unlocks the
> long-horizon adaptation we've already demonstrated end-to-end. Thanks.

---

## Pre-recording prep checklist

- [ ] Render `fig/ARCHITECTURE.md` Mermaid block to `fig/architecture_diagram.png`
      via <https://mermaid.live> or `mmdc`.
- [ ] Run `python notebooks/kaggle-simsat-lfm-v3/fetch_run_outputs.py` to
      ensure `.kaggle_output_v3/holdout_eval_report.json` is fresh on disk
      for the terminal screenshot.
- [ ] Open `submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png`
      in a clean image viewer.
- [ ] Open `SUBMISSION_CASEBOOK.md` Rotterdam section in a markdown viewer.
- [ ] Open `.kaggle_output/ttt_proof_of_life_receipt.json` formatted nicely.
- [ ] Test mic levels — 30s test recording first.
- [ ] Close all notification sources (Slack, mail, etc.).
- [ ] Set screen resolution to 1920×1080 to match output.

## Recording approach options

**Option 1 — single take voiceover with shot list:**
- Set up OBS Studio with two scenes: full-screen terminal, full-screen browser.
- Record voiceover and screen as one continuous take.
- Edit cuts in DaVinci Resolve (free).

**Option 2 — voiceover-first, then b-roll:**
- Record voiceover audio in Audacity (free) — gives you a polished clean track.
- Record screen captures separately for each shot.
- Layer voiceover over b-roll in DaVinci Resolve.
- More polished result, ~1-2 hours longer to produce.

**Option 3 — use Loom (one-button):**
- Loom does voiceover + screen + face-cam in one click.
- Less polish, but ships in 30 minutes if you're confident in the take.

For a hackathon with a 17:00 PDT deadline, **Option 1 or 3** is the right
call. Save Option 2 for if you finish v5 + the form well before the deadline.

## Word-counts and pacing

- ~440 words at a comfortable 115 wpm = 3 min 50 sec.
- If you naturally talk at 130 wpm, target ~500 words; pad with 5-second
  silent transitions between shots so the viewer can absorb the on-screen
  numbers.
- Don't rush the headline numbers (shot 4) — those are the biggest rubric
  hit. Slow down for "zero point eight four four."

## Caption / closed-caption file

Auto-generate captions from the video itself in DaVinci Resolve (Studio
license; the free version doesn't have it) or in YouTube Studio (free) by
uploading and downloading the auto-generated `.srt`. The voiceover is clean
and short, so auto-captions will be ~95% accurate; budget 15 minutes for a
human pass.

Submitting captions with the video is a small accessibility +1 that judges
notice.
