#!/usr/bin/env python3
"""Overlay your voiceover onto silent_demo.mp4 -> simsat_demo_final.mp4.

Usage:
    python video_assets/overlay_audio.py path/to/voiceover.{wav,mp3,m4a}

Optional flags:
    --offset SECONDS   Shift the audio start (positive = audio starts later)
    --music PATH       Add a low-volume background music track underneath VO
    --music-vol 0.08   Background music volume (0.0-1.0, default 0.08)

Recording tips:
- Match the timing in VIDEO_SCRIPT.md when you read.
- Aim for ~440 words at ~115 wpm = 3 min 50 sec.
- Record in a quiet room, use Audacity to remove silence at start/end + apply
  -3 dB compression.
- WAV is best (lossless), but MP3/M4A also work fine.
- If your VO is shorter than 230 s, the video freezes on the close card.
- If longer than 230 s, the video gets clipped — re-record or split the long
  shot into a longer hold.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

ROOT = Path(__file__).resolve().parents[1]
VIDEO_IN = ROOT / "video_assets" / "silent_demo.mp4"
OUT = ROOT / "video_assets" / "simsat_demo_final.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voiceover", type=Path, help="Path to your voiceover audio (wav/mp3/m4a)")
    ap.add_argument("--offset", type=float, default=0.0, help="Shift audio start (seconds)")
    ap.add_argument("--music", type=Path, default=None, help="Optional background music file")
    ap.add_argument("--music-vol", type=float, default=0.08)
    ap.add_argument("--out", type=Path, default=OUT, help="Output path")
    args = ap.parse_args()

    if not VIDEO_IN.exists():
        raise FileNotFoundError(f"Run build_video.py first: {VIDEO_IN}")
    if not args.voiceover.exists():
        raise FileNotFoundError(args.voiceover)

    video = VideoFileClip(str(VIDEO_IN))
    print(f"Video:    {video.duration:.1f}s, {video.size}, {video.fps} fps")

    vo = AudioFileClip(str(args.voiceover))
    print(f"VO:       {vo.duration:.1f}s ({args.voiceover.name})")
    if args.offset:
        vo = vo.with_start(args.offset)
        print(f"  offset: +{args.offset}s")

    audio_tracks = [vo]
    if args.music and args.music.exists():
        music = AudioFileClip(str(args.music)).with_volume_scaled(args.music_vol)
        # Loop or cut music to video length
        if music.duration < video.duration:
            from moviepy.audio.fx import AudioLoop
            music = music.with_effects([AudioLoop(duration=video.duration)])
        else:
            music = music.subclipped(0, video.duration)
        audio_tracks.append(music)
        print(f"Music:    {args.music.name} at vol {args.music_vol}")

    final_audio = CompositeAudioClip(audio_tracks) if len(audio_tracks) > 1 else audio_tracks[0]
    final = video.with_audio(final_audio)

    print(f"\nWriting {args.out} ...")
    final.write_videofile(
        str(args.out),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nDone: {args.out}  ({size_mb:.1f} MB)")
    if size_mb > 50:
        print("  Note: > 50 MB — many submission forms cap at 50 MB.")
        print("        Try preset=fast or a lower bitrate if needed.")


if __name__ == "__main__":
    main()
