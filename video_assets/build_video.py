#!/usr/bin/env python3
"""Stitch the 9 rendered frames into a silent demo video.

Timing matches VIDEO_SCRIPT.md shot list:
  shot 1:  0:00-0:08   8s   title cold open
  shot 2:  0:08-0:20   12s  title + Rotterdam band
  shot 3:  0:20-0:50   30s  architecture diagram
  shot 4:  0:50-1:20   30s  v3 holdout table (headline)
  shot 5:  1:20-1:35   15s  MAE drop callout
  shot 6:  1:35-2:30   55s  Rotterdam case panel
  shot 7:  2:30-3:05   35s  TTT receipt
  shot 8:  3:05-3:30   25s  v4 negative result
  shot 9:  3:30-3:50   20s  close card
                       ────
                       230s = 3 min 50 sec

Output: video_assets/silent_demo.mp4 (1920x1080, 30fps, h.264)
"""
from __future__ import annotations
from pathlib import Path

from moviepy import ImageClip, concatenate_videoclips, CompositeVideoClip

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "video_assets" / "frames"
OUT = ROOT / "video_assets" / "silent_demo.mp4"

# (shot_idx, duration_seconds)
SHOTS = [
    (1,  8),
    (2, 12),
    (3, 30),
    (4, 30),
    (5, 15),
    (6, 55),
    (7, 45),  # was 35; extended for two-class TTT lift voiceover
    (8, 25),
    (9, 25),  # was 20; extended for "JSON-schema refinement" framing in close
]


def main():
    clips = []
    for idx, dur in SHOTS:
        path = FRAMES / f"shot{idx}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        # Add a half-second crossfade between shots for visual smoothness.
        clip = ImageClip(str(path)).with_duration(dur)
        clips.append(clip)
        print(f"  shot{idx}: {dur}s  ({path.name})")

    # Simple concatenation (no fades — keeps the file simple to overlay audio on)
    video = concatenate_videoclips(clips, method="compose")
    print(f"\nTotal duration: {video.duration:.1f} s")
    print(f"Writing {OUT} ...")
    video.write_videofile(
        str(OUT),
        fps=30,
        codec="libx264",
        audio=False,
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p"],  # widely-compatible pixel format
    )
    print(f"\nDone: {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
