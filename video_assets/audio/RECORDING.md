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

> I'm Ben Haslam, founder of the **HumanAI Convention**

HumanAI Convention is a humble project to enable mass human flourishing through robust, ethical training data.

The hypothesis is that the synthetic data AI scales on must be balanced with data from human lived experience, and an ethical framework — humans living their best lives — provides the optimal data.

That's the **Viability Condition, or Convention,** that underlies this project.

**Visual:** black → phi-with-dot mark fades in white over 2 s, then the
"Human AI / Convention" wordmark crossfades in below at ~5 s. Both hold
to the end.

**Land carefully:** "Viability Convention" — that's the term you're
introducing. Slow it down.

---

### 🎙 shot2.wav — 28 s — operationalization bridge

This proposal operationalizes that hypothesis through SimSat — a working prototype where the Convention's six non-compensatory viability gates govern continual learning on a real-world problem. I chose satellite encounter triage because the constraints are physical: minutes between decisions, no round-trip to ground, an uplink too small for weight updates but large enough for compact operator labels. The architecture is the contribution; the satellite is the test case.

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

### 🎙 shot4.wav — 28 s — 

Two tracks, one architecture. The Liquid Track runs LFM 2.5-VL 450M. The General AI Track runs Gemma 4 E2B — and is open to any vision-language model that emits the eight-key JSON contract. Same scaffold, same six viability gates, same operator-curated stream across both. The backend is the variable; the architecture is what generalizes.


---

shot5.wav headline


The Headline is: we fine-tuned the LFM encoder on a hundred sixty-five operator-reviewed Sentinel-2 tiles — each one a human judgment encoded as training data. Base model exact action agreement on a balanced thirty-two-row holdout, demonstrating a 68.8 percent improvement.

---

### 🎙 shot6.wav — 14 s — MAE drop

Score Mean Absolute Error, or MAE — the alignment between model and operator judgment — drops from 0.365 to 0.055. The model and the human now agree to within one band step. Per-class accept and refine are perfect. The adapter is on Hugging Face under an Apache 2.0 license.

---

### 🎙 shot7.wav — 55 s — Rotterdam case (longest)

Rotterdam, 48.8 percent cloud cover. Scaffold says accept — geometry is good. Trust layer flips to refine — the cloud might hide containers. The operator confirms accept: the visible fifty-one percent of the basin was operationally enough. That human judgment is the signal. Every operator review is one of these — the calibration loop runs on every window.

---

### 🎙 shot8.wav — 50 s — TTT class-targeted lift

Vision-Language-Action or VLA-layer test-time training under operator-curated stream — each encounter a human judgment, every update gate-filtered. Stability first: fifty cycles on the v three adapter, zero divergence, parse rate one-point-zero throughout. Then the architectural claim. Sixteen skip-class encounters: skip lifted from 0.375 to 0.57. Sixteen defer-class encounters: defer lifted from 0.25 to 0.875 — plus seventy-five points. These numbers are the proof of mechanism. The mechanism is operator participation becoming model improvement, gate-filtered, on a real LFM checkpoint.

---

### 🎙 shot9.wav — 25 s — three honest negatives

Three honest negatives. More imbalanced data — minus 6.33. More class-balanced data — minus 15.6. Different recipe — minus 3.1. Same conclusion across three independent angles: v three sits at a local optimum that offline data cannot escape. Static training has hit its ceiling here; the next axis is operator participation — runtime contributions, gate-filtered, becoming gradient signal. All three negatives are published in the methodology doc.

---

### 🎙 shot10.wav — 22 s — close

What this loop enables is mass human flourishing through ethical training data — one operator review at a time, gated by six viability checks, contributed locally where lived experience exists. SimSat is one instance. The architecture moves to medicine, education, anywhere expert humans guide AI. Apache licensed 2.0 weights, AGPL 3 code, public kernels. This framework is built for the Orin 16 gig system. Thanks.

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
