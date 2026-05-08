#!/usr/bin/env python3
"""Stitch the 9 rendered frames into a silent demo video using ffmpeg directly.

Much faster than moviepy compose for a video of static images: each shot is
a single PNG looped for its duration, with H.264 ultrafast encode. Total wall
time on i5-9600K is well under a minute for the whole 4:43 demo.

Timing matches VIDEO_SCRIPT.md shot list:
  shot 1: 30s  HumanAI Convention logo on black (fade-in 2s)
              Founder voiceover plays over this.
  shot 2: 28s  SimSat title (Rotterdam Sentinel band)
              Bridge voiceover ("This proposal operationalizes...").
  shot 3: 30s  architecture diagram
  shot 4: 30s  v3 holdout table (headline)
  shot 5: 15s  MAE drop callout
  shot 6: 55s  Rotterdam case panel
  shot 7: 45s  TTT receipt — class-targeted lifts
  shot 8: 25s  v4/v5/v3+ negative results
  shot 9: 25s  close card (humanaiconvention.com first)
              ────
              283s = 4 min 43 sec

Output: video_assets/silent_demo.mp4 (1920x1080, 30fps, h.264, yuv420p)
"""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "video_assets" / "frames"
OUT = ROOT / "video_assets" / "silent_demo.mp4"
SEGS = ROOT / "video_assets" / "segments"
SEGS.mkdir(exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 30

# (shot_idx, duration_seconds)
SHOTS = [
    (1, 30),  # HAIC logo on black, fade-in 2s; founder voiceover
    (2, 28),  # SimSat title; bridge voiceover
    (3, 30),  # architecture diagram
    (4, 30),  # v3 holdout headline
    (5, 15),  # MAE drop callout
    (6, 55),  # Rotterdam case
    (7, 45),  # TTT receipts
    (8, 25),  # honest negatives
    (9, 25),  # close card
]


def encode_segment(idx: int, dur: int) -> Path:
    """Encode one PNG → mp4 segment of given duration.

    Shot 1 is a special case: two PNGs (mark only + mark with wordmark)
    sequenced via xfade so the wordmark appears as the founder says
    "HumanAI Convention" (~5s into the voiceover).

      0-2s   black → mark fades in
      2-4s   mark holds
      4-5s   crossfade from mark to mark+wordmark (1s)
      5-end  mark+wordmark holds
    """
    dst = SEGS / f"seg{idx}.mp4"

    if idx == 1:
        mark = FRAMES / "shot1.png"
        full = FRAMES / "shot1_full.png"
        if not mark.exists() or not full.exists():
            raise FileNotFoundError(f"need both {mark.name} and {full.name}")
        first_dur = 5
        second_dur = dur - 4  # second clip starts at xfade offset (4s)
        cmd = [
            FFMPEG, "-y",
            "-loop", "1", "-framerate", str(FPS), "-t", str(first_dur), "-i", str(mark),
            "-loop", "1", "-framerate", str(FPS), "-t", str(second_dur), "-i", str(full),
            "-filter_complex",
            f"[0:v]fade=t=in:st=0:d=2[a];"
            f"[a][1:v]xfade=transition=fade:duration=1:offset=4,format=yuv420p[v]",
            "-map", "[v]",
            "-t", str(dur),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-r", str(FPS),
            str(dst),
        ]
    else:
        src = FRAMES / f"shot{idx}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        cmd = [
            FFMPEG, "-y",
            "-loop", "1",
            "-framerate", str(FPS),
            "-t", str(dur),
            "-i", str(src),
            "-vf", "format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-r", str(FPS),
            str(dst),
        ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def main() -> None:
    print(f"ffmpeg: {FFMPEG}")
    segs = []
    for idx, dur in SHOTS:
        seg = encode_segment(idx, dur)
        size_kb = seg.stat().st_size / 1024
        print(f"  seg{idx}: {dur:>3}s  ({size_kb:>6.1f} KB)")
        segs.append(seg)

    # Concat with the concat demuxer (lossless — no re-encode)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for seg in segs:
            f.write(f"file '{seg.as_posix()}'\n")
        list_path = Path(f.name)

    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(OUT),
    ]
    print(f"\nconcat -> {OUT.name}")
    subprocess.run(cmd, check=True, capture_output=True)
    list_path.unlink()

    total = sum(d for _, d in SHOTS)
    print(f"\nTotal duration: {total}s ({total // 60}:{total % 60:02d})")
    print(f"Done: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
