"""Collect one primary video per topic, with lossless audio and source provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        candidates = sorted(p for p in folder.glob("*.mp4") if p.with_suffix(".srt").is_file())
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one main MP4/SRT pair in {folder}: {candidates}")
        video = candidates[0]
        subtitle = video.with_suffix(".srt")
        info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)
        ], text=True, encoding="utf-8"))
        streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
        if len(streams) != 1:
            raise ValueError(f"Review audio stream selection for {video}: {streams}")
        key = folder.name.replace(" ", "_")
        audio = destination / f"{key}.flac"
        srt = audio.with_suffix(".srt")
        duration = float(info["format"]["duration"])
        if audio.exists():
            previous = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio)
            ], text=True))
            if abs(float(previous["format"]["duration"]) - duration) > 0.25:
                raise ValueError(f"Existing audio has wrong duration: {audio}")
        else:
            temporary = audio.with_suffix(".partial.flac")
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "24000",
                "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24",
                "-compression_level", "5", str(temporary)
            ], check=True)
            temporary.replace(audio)
        digest = hashlib.sha256(subtitle.read_bytes()).hexdigest()
        if srt.exists() and hashlib.sha256(srt.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing subtitle differs: {srt}")
        if not srt.exists():
            shutil.copy2(subtitle, srt)
        row = {
            "topic": folder.name, "source_video": str(video), "source_subtitle": str(subtitle),
            "source_bytes": video.stat().st_size, "source_mtime_ns": video.stat().st_mtime_ns,
            "duration_s": duration, "audio": audio.name, "subtitle": srt.name,
            "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "subtitle_sha256": digest, "source_audio_stream": streams[0],
        }
        rows.append(row)
        print(f"Collected {len(rows)}: {folder.name} ({duration / 60:.1f} min)", flush=True)
        report = {
            "source_root": str(source), "selection": "one immediate-child main MP4 with exact matching SRT per topic; auxiliary media excluded",
            "audio_format": "mono 24 kHz 24-bit FLAC, no denoising or level changes",
            "sources": rows, "total_duration_s": sum(r["duration_s"] for r in rows),
        }
        temporary_manifest = destination / "sources.json.tmp"
        temporary_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_manifest.replace(destination / "sources.json")


if __name__ == "__main__":
    main()
