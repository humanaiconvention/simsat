#!/usr/bin/env python3
"""Assemble final demo video from silent_demo.mp4 + nine voiceover WAVs.

Reads video_assets/audio/shot{1..9}.wav. Each one is normalized to
mono 48 kHz, padded with trailing silence (or truncated) to its allotted
shot duration, concatenated to a single voiceover track, and muxed onto
silent_demo.mp4 to produce simsat_demo_final.mp4.

Usage:
    python video_assets/assemble_voiceover.py
    python video_assets/assemble_voiceover.py --check     # only show timing
    python video_assets/assemble_voiceover.py --music path/to/bg.mp3 --music-vol 0.06

Missing wav files are gracefully replaced with pure silence — useful
while recording iteratively. Each missing shot is logged so you can see
which segments are still pending.

Recording tips:
- Save each take as video_assets/audio/shot{N}.wav (1..9).
- WAV is best (lossless). MP3/M4A also accepted — change extension below.
- Aim for the target duration but don't sweat ±2 s; this script pads/cuts
  to fit. If you go SIGNIFICANTLY long on a shot you'll feel it cut off
  mid-word — re-record and trim the recording, or extend the silent
  shot's duration in build_video.py and rebuild.
"""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "video_assets" / "audio"
SILENT = ROOT / "video_assets" / "silent_demo.mp4"
DEFAULT_OUT = ROOT / "video_assets" / "simsat_demo_final.mp4"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 48000

# Match SHOTS in build_video.py — (shot_idx, duration_seconds)
SHOTS = [
    (1, 30),  # HAIC logo cold open + founder voiceover
    (2, 28),  # SimSat title bridge
    (3, 30),  # architecture diagram
    (4, 30),  # v3 holdout headline
    (5, 15),  # MAE drop callout
    (6, 55),  # Rotterdam case
    (7, 45),  # TTT receipts
    (8, 25),  # honest negatives
    (9, 25),  # close card
]


def probe_duration(wav: Path) -> float:
    """Return duration in seconds of a wav/audio file via ffmpeg."""
    proc = subprocess.run(
        [FFMPEG, "-i", str(wav), "-hide_banner", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # ffmpeg writes duration to stderr like "  Duration: 00:00:14.32, ..."
    for line in proc.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            ts = line.split(",")[0].split("Duration:")[1].strip()
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return -1.0


def normalize_segment(idx: int, dur: int, ext_priority=("wav", "mp3", "m4a", "flac")) -> tuple[Path, str, float]:
    """Normalize shot{idx}.* to mono 48 kHz, pad/trim to dur seconds.

    Returns (output_path, status_str, source_duration_s).
    status_str is one of: "recorded", "silence" (for graceful skip).
    """
    src = None
    for ext in ext_priority:
        candidate = AUDIO / f"shot{idx}.{ext}"
        if candidate.exists():
            src = candidate
            break

    dst = Path(tempfile.gettempdir()) / f"haic_seg{idx}.wav"

    if src is None:
        # Generate pure silence of the target duration
        cmd = [
            FFMPEG, "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
            "-t", str(dur),
            str(dst),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return dst, "silence", 0.0

    src_dur = probe_duration(src)

    # Resample to mono 48 kHz, pad with infinite silence, then trim to exactly dur
    cmd = [
        FFMPEG, "-y",
        "-i", str(src),
        "-af", (
            f"aresample={SAMPLE_RATE}"
            ",aformat=channel_layouts=mono"
            ",apad"
            f",atrim=0:{dur}"
            ",asetpts=N/SR/TB"
        ),
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst, "recorded", src_dur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only report what's recorded vs missing — don't build")
    ap.add_argument("--music", type=Path, default=None,
                    help="optional background music file (mp3/wav)")
    ap.add_argument("--music-vol", type=float, default=0.06,
                    help="background music volume 0.0-1.0 (default 0.06)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output mp4 path (default {DEFAULT_OUT.name})")
    args = ap.parse_args()

    AUDIO.mkdir(exist_ok=True, parents=True)
    if not SILENT.exists():
        raise FileNotFoundError(f"missing {SILENT} — run build_video.py first")

    print(f"audio dir: {AUDIO}")
    print(f"target:    {args.out}")
    print()

    # Status report first
    total_recorded = 0
    total_target = 0
    for idx, dur in SHOTS:
        src = next(
            (AUDIO / f"shot{idx}.{ext}" for ext in ("wav", "mp3", "m4a", "flac")
             if (AUDIO / f"shot{idx}.{ext}").exists()),
            None,
        )
        target = dur
        total_target += target
        if src is not None:
            actual = probe_duration(src)
            status = "OK" if abs(actual - target) <= 2 else (
                "LONG" if actual > target else "SHORT"
            )
            print(f"  shot{idx}: {actual:5.1f}s recorded / {target:>3}s target  [{status}]  {src.name}")
            total_recorded += 1
        else:
            print(f"  shot{idx}:    ---  /  {target:>3}s target  [PENDING]")
    print(f"\n  {total_recorded}/{len(SHOTS)} shots recorded; total target {total_target}s "
          f"({total_target // 60}:{total_target % 60:02d})")

    if args.check:
        return

    if total_recorded == 0:
        print("\nNo recordings yet — run with --check after recording.")
        return

    # Normalize and concat
    print("\nNormalizing segments...")
    segments = []
    for idx, dur in SHOTS:
        seg, status, src_dur = normalize_segment(idx, dur)
        marker = "🎙" if status == "recorded" else "—"
        delta = f"({src_dur:+.1f}s)" if status == "recorded" else ""
        print(f"  {marker} seg{idx}: padded to {dur}s {delta}")
        segments.append(seg)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for seg in segments:
            f.write(f"file '{seg.as_posix()}'\n")
        list_path = Path(f.name)

    voiceover = Path(tempfile.gettempdir()) / "haic_voiceover.wav"
    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(voiceover),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    list_path.unlink()
    print(f"\nvoiceover: {voiceover.stat().st_size / 1024:.0f} KB")

    # Mux voiceover (and optional music) onto silent video
    print(f"\nMuxing onto {SILENT.name} ...")
    if args.music and args.music.exists():
        # Mix VO + music with sidechain ducking so music dips when VO speaks
        cmd = [
            FFMPEG, "-y",
            "-i", str(SILENT),
            "-i", str(voiceover),
            "-i", str(args.music),
            "-filter_complex",
            f"[2:a]volume={args.music_vol},aloop=loop=-1:size=2e9[bgm];"
            f"[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[out]",
            "-map", "0:v",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(args.out),
        ]
        print(f"  + music: {args.music.name} @ vol {args.music_vol}")
    else:
        cmd = [
            FFMPEG, "-y",
            "-i", str(SILENT),
            "-i", str(voiceover),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v",
            "-map", "1:a",
            "-shortest",
            str(args.out),
        ]
    subprocess.run(cmd, check=True, capture_output=True)

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nDone: {args.out}  ({size_mb:.1f} MB)")
    if size_mb > 50:
        print("  Note: > 50 MB — many submission forms cap at 50 MB.")


if __name__ == "__main__":
    main()
