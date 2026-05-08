# Video Recording Assets — pre-staged for VIDEO_SCRIPT.md

Everything in this directory is ready to display on screen during the demo
recording. Each file maps to a specific shot in [`../VIDEO_SCRIPT.md`](../VIDEO_SCRIPT.md).

| Shot | File | Show this on screen during voiceover |
|---|---|---|
| 1 | (title card — make in slide tool) | "SimSat — On-Orbit AI for Satellite Encounter Tasking" + Rotterdam thumbnail blurred |
| 3 | [`../fig/architecture_diagram.png`](../fig/architecture_diagram.png) | Mermaid-rendered architecture (1449×1684 PNG) |
| 4 | [`shot4_v3_holdout_table.txt`](shot4_v3_holdout_table.txt) | v3 holdout table — base vs tuned, all metrics + per-class |
| 5 | (same as shot 4) | Hold the table, just zoom to the score_mae row |
| 6 | [`shot6_rotterdam_panel.txt`](shot6_rotterdam_panel.txt) + [`../submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png`](../submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png) | Rotterdam case panel beside the Sentinel image |
| 7 | [`shot7_ttt_receipt.txt`](shot7_ttt_receipt.txt) | TTT proof-of-life receipt summary |
| 8 | [`shot8_v3_v4_v5_negative.txt`](shot8_v3_v4_v5_negative.txt) | v3/v4 comparison + v5 in-flight note |
| 9 | [`shot9_close.txt`](shot9_close.txt) | Final card: code/weights/kernels/datasets URLs |

## How to display these on screen

These are **ASCII text panels** designed to look good on a terminal screen
recording. Two ways to display:

**Option 1 — full-screen terminal (recommended):**
```bash
# Open a terminal at 14pt monospace, full-screen, dark or light theme
clear && cat video_assets/shot4_v3_holdout_table.txt
# Hold the frame for the voiceover beat. Then:
clear && cat video_assets/shot6_rotterdam_panel.txt
# etc.
```

**Option 2 — markdown viewer (also works well):**
- Convert each `.txt` to `.md` with a code-block fence and view in any
  markdown previewer.

For the architecture frame (shot 3), display
[`../fig/architecture_diagram.png`](../fig/architecture_diagram.png) directly
in an image viewer or browser.

For the Rotterdam frame (shot 6), arrange two windows side-by-side:
- Left half: the Sentinel image
- Right half: the ASCII panel in a terminal

## Tomorrow morning checklist (recap)

When you're ready to record:

1. ☐ Pull v5 results when they land (~01:30 PDT). If v5 ≥ v3, update the
   v3 numbers in `shot4_v3_holdout_table.txt` to v5 (or add a v5 row).
2. ☐ Open all four `shot*.txt` files in terminal tabs.
3. ☐ Open `fig/architecture_diagram.png` and the Rotterdam Sentinel image.
4. ☐ Run mic check.
5. ☐ Close notifications.
6. ☐ Set screen to 1920×1080.
7. ☐ Record per `VIDEO_SCRIPT.md`. ← human required from here
8. ☐ Edit + caption + export 1080p mp4.
9. ☐ Upload to YouTube unlisted (or the form's video field directly).
10. ☐ Submit form using `SUBMISSION_ABSTRACT.md` content.

Steps 7-10 are inherently human work. Everything before that is staged.

---

## NEW: pre-built silent video + audio overlay tool

The shot panels are also pre-rendered into a single 1920×1080 silent video
at the timing in `VIDEO_SCRIPT.md`:

```bash
python video_assets/build_video.py
# Produces video_assets/silent_demo.mp4 (4.4 MB, 3:50, 30 fps)
```

To add your voiceover:

```bash
# Record voiceover in Audacity (or any DAW) following VIDEO_SCRIPT.md.
# Save as voiceover.wav (or .mp3/.m4a) anywhere.

python video_assets/overlay_audio.py path/to/voiceover.wav
# Produces video_assets/simsat_demo_final.mp4 — the submission file
```

**Optional background music:** add `--music path/to/track.mp3 --music-vol 0.08`.

**Adjust audio start:** add `--offset 1.5` (positive = VO starts later, e.g. if
your recording has 1.5s of silence to trim manually).

**Re-render the silent video** (after updating numbers in shot frames):

```bash
# 1. Edit video_assets/render_frames.py if numbers changed (e.g. v5 lands)
# 2. Re-render frames:
python video_assets/render_frames.py
# 3. Rebuild silent video:
python video_assets/build_video.py
# 4. Re-overlay your audio:
python video_assets/overlay_audio.py path/to/voiceover.wav
```

The MP4 outputs are gitignored (regenerable). Frames PNGs also gitignored.
The build/overlay/render scripts and shot text panels are committed.

