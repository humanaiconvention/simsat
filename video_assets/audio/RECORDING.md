# Voiceover recording — segmented plan

Record each segment as `shot{N}.wav` in this folder. Then run:

```bash
python video_assets/assemble_voiceover.py --check     # status report
python video_assets/assemble_voiceover.py             # build final mp4
```

The script pads each take with trailing silence to the shot's exact target
duration, so you don't have to hit the timing precisely — aim within ±2 s.

## Recording order recommendation

Warm up with shorter, less emotionally loaded takes first. Save shot 1
for last when your voice is fully dialed in.

| # | When to record | Length |
|---|---|---|
| 5 | First (warm-up — just numbers) | 14 s |
| 9 | Second (sets the close register) | 22 s |
| 4 | Third | 28 s |
| 3 | | 28 s |
| 8 | | 25 s |
| 2 | | 28 s |
| 7 | | 50 s |
| 6 | | 55 s |
| 1 | **Last** (founder identity, fully warmed up) | 30 s |

## Setup

- **Audacity** is the easiest recorder (free).
- Save as 48 kHz mono WAV. (The script normalizes anyway, but starting
  clean = less to fix later.)
- Stand up. Glass of water. Phone on Do Not Disturb.
- 30-second silent room-tone take saved as `room_tone.wav` is useful if
  you need to splice in a breath or fix a click.

---

## The scripts

Each block below is what you read for that segment. Punctuation cues
intentional pauses. *Italics* mark words to lean on a little.

---

### 🎙 shot1.wav — 30 s — founder identity

> I'm Ben Haslam, founder of the **HumanAI Convention** — a humble project
> to enable mass human flourishing through robust, ethical training data.
>
> The HumanAI Convention's hypothesis is that the synthetic data AI scales
> on must be balanced with data from human lived experience, and an
> ethical framework — humans living their best lives — provides the
> optimal data.
>
> That's the **Viability Convention** as I hypothesize it.

**Visual:** black → phi-with-dot mark fades in white over 2 s, then the
"Human AI / Convention" wordmark crossfades in below at ~5 s. Both hold
to the end.

**Land carefully:** "Viability Convention" — that's the term you're
introducing. Slow it down.

---

### 🎙 shot2.wav — 28 s — operationalization bridge

> This proposal takes that hypothesis and operationalizes it through
> SimSat — a working prototype where the Convention's six non-compensatory
> viability gates govern continual learning on a real-world problem.
>
> We chose satellite encounter triage because the constraints are
> physical: minutes between decisions, no round-trip to ground, an uplink
> too small to ferry weight updates.
>
> *What it can ferry is JSON.*
>
> The architecture is the contribution; the satellite is the test case.

**Visual:** SimSat title card from Claude Design — phi-with-dot lockup
top-left, "SimSat." centered, tagline, Sentinel band of Rotterdam.

**Land carefully:** Pause before *"What it can ferry is JSON."* That's
the rhetorical peak of the whole video.

---

### 🎙 shot3.wav — 28 s — architecture overview

> One vision-language tile encoder — Liquid AI's
> **LFM-two-point-five-VL-450M** — feeds a planner over encounter windows.
>
> A trust layer scores each candidate against geometry, priority, cloud,
> and visibility.
>
> Six viability gates govern any continual learning that runs on top.

**Visual:** Architecture diagram (top half).

**Pronunciation:** Don't rush "L-F-M two point five V-L four-fifty M."
Say each piece distinctly.

---

### 🎙 shot4.wav — 28 s — headline numbers

> Headline: we fine-tuned the LFM encoder with LoRA on 165
> operator-reviewed Sentinel-2 tiles.
>
> Base model exact action agreement on a balanced 32-row holdout: 0.156.
>
> Tuned: **zero point eight four four**.
>
> Plus sixty-eight point eight points.

**Visual:** v3 holdout report table.

**Land carefully:** Slow down for "zero point eight four four." This is
the biggest rubric hit in the video. Don't rush past it.

---

### 🎙 shot5.wav — 14 s — MAE drop

> Score MAE drops from 0.365 to 0.055 — about a thirty-one-point reduction
> in band-mapped prediction error.
>
> Per-class accept and refine are perfect.
>
> The adapter is on Hugging Face under Apache-2.0.

**Visual:** MAE drop callout.

**Tone:** Crisp. Don't oversell. Numbers stand on their own.

---

### 🎙 shot6.wav — 55 s — Rotterdam case (longest)

> Rotterdam, forty-eight point seven eight percent cloud cover.
>
> The scaffold says accept — geometry is good.
>
> The trust layer disagrees, flips it to refine — the cloud might hide
> containers.
>
> The operator confirms accept: the visible fifty-one percent of the basin
> was operationally enough.
>
> This is the calibration loop.
>
> The same loop runs on every window.

**Visual:** Rotterdam Sentinel image full-frame + case panel showing
scaffold → trust → operator boxes.

**Tone:** Tell it like a story. The scaffold-trust-operator handoff is
the dramatic beat. You can take your time on this one — there's 55
seconds of visual.

---

### 🎙 shot7.wav — 50 s — TTT class-targeted lift

> VLA-layer test-time training, validated end-to-end.
>
> Stability first: fifty steps on the v three adapter, forty-eight applied,
> zero divergence, parse rate one-point-zero throughout.
>
> Then the headline result, on two classes.
>
> Sixteen skip-class encounters streamed: skip lifted from
> zero-point-three-seven-five to zero-point-seven-five — plus thirty-seven
> points.
>
> Sixteen defer-class encounters streamed: defer lifted from
> zero-point-one-two-five to zero-point-eight-seven-five — plus
> seventy-five points.
>
> Average lift across two classes, plus fifty-six points in sixteen steps.
>
> TTT under operator-curated stream empirically lifts target-class accuracy
> per pass on **TWO independent classes** — the architectural claim,
> validated on a real LFM checkpoint.

**Visual:** TTT receipts panel.

**Breath points:** after "parse rate one-point-zero throughout" and after
"plus seventy-five points." This is the densest segment — give yourself
two breaths.

---

### 🎙 shot8.wav — 25 s — three honest negatives

> We tried three different ways to beat v three.
>
> More imbalanced data: minus six point three.
>
> More class-balanced data: minus fifteen point six.
>
> A different recipe — lower learning rate, longer schedule, more dropout
> — minus three point one.
>
> Three independent angles, same conclusion: v three sits at a local
> optimum on this architecture and this holdout.
>
> That's empirical evidence for the runtime-TTT lane, not just an
> architectural claim. All three negatives are published in the
> methodology doc.

**Visual:** v3 / v4 / v5 / v3+ comparison panel.

**Tone:** Don't apologize for the minuses. They are the evidence — that's
the whole point. Read them flat and confident.

---

### 🎙 shot9.wav — 22 s — close

> On-orbit inference, continually refined by operator-labelled JSON within
> uplink bandwidth parameters.
>
> Apache-2.0 weights on Hugging Face. Public training kernels on Kaggle.
> AGPL-3 code on GitHub.
>
> Built for the prize hardware — Orin sixteen gig is what unlocks the
> long-horizon adaptation we've already demonstrated end-to-end.
>
> Thanks.

**Visual:** Closing card from Claude Design — humanaiconvention.com,
URL stack, Orin line.

**Tone:** The exhale. Conversational. "Thanks" should feel earned, not
perfunctory.

---

## After all 9 are recorded

```bash
cd D:\SimSat
python video_assets/assemble_voiceover.py --check    # status report
python video_assets/assemble_voiceover.py            # build final mp4
```

Output: `video_assets/simsat_demo_final.mp4`.

If anything sounds off, re-record just that one segment and re-run.

## If you want background music

Drop a track at e.g. `D:\Music\bg.mp3` and:

```bash
python video_assets/assemble_voiceover.py --music D:\Music\bg.mp3 --music-vol 0.06
```

The script mixes it under the voiceover at low volume. 0.06 is a safe
default — bump to 0.08 if you can't hear it, drop to 0.04 if it competes
with your voice.
