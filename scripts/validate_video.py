#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the actual release-video streams and its committed manifest.

This check deliberately uses ``ffprobe`` instead of trusting filename
extensions or values written by the renderer.  It is shared by the local
builder and the publication workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image

EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
MIN_DURATION_SECONDS = 180.0
MAX_DURATION_SECONDS = 300.0
MAX_CAPTION_LINES = 3
SRT_END_TOLERANCE_SECONDS = 0.05
STREAM_DURATION_TOLERANCE_SECONDS = 0.10
MIN_MEAN_VOLUME_DB = -45.0
MIN_PEAK_VOLUME_DB = -20.0
_SRT_TIMING = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> "
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"video is missing or empty: {path}")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def probe_audio_levels(path: Path) -> dict[str, float]:
    """Measure actual decoded audio so a silent/truncated track cannot pass."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    levels: dict[str, float] = {}
    for key in ("mean_volume", "max_volume"):
        match = re.search(rf"{key}:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB", result.stderr)
        if match is None:
            raise ValueError(f"ffmpeg did not report {key} for the narration")
        levels[f"{key}_db"] = float(match.group(1))
    return levels


def validate_audio_signal(path: Path) -> dict[str, float]:
    levels = probe_audio_levels(path)
    if levels["mean_volume_db"] < MIN_MEAN_VOLUME_DB:
        raise ValueError("narration mean level is effectively silent")
    if levels["max_volume_db"] < MIN_PEAK_VOLUME_DB:
        raise ValueError("narration peak level is effectively silent")
    return levels


def _srt_seconds(groups: tuple[str, str, str, str]) -> float:
    hours, minutes, seconds, milliseconds = (int(value) for value in groups)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("SRT timestamp minutes and seconds must be below 60")
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def validate_srt(captions: Path, media_duration: float) -> dict[str, Any]:
    """Validate cue structure, ordering, bounds, and the visual line budget."""
    if not captions.is_file() or captions.stat().st_size == 0:
        raise ValueError(f"captions are missing or empty: {captions}")
    if not isinstance(media_duration, (int, float)) or media_duration <= 0:
        raise ValueError("caption validation requires a positive media duration")
    text = captions.read_text(encoding="utf-8-sig").strip()
    blocks = re.split(r"\r?\n\s*\r?\n", text) if text else []
    if not blocks:
        raise ValueError("captions contain no SRT cues")

    previous_end = 0.0
    max_lines = 0
    for expected_index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"SRT cue {expected_index} has no visible caption text")
        try:
            cue_index = int(lines[0])
        except ValueError as exc:
            raise ValueError(f"SRT cue {expected_index} has a nonnumeric index") from exc
        if cue_index != expected_index:
            raise ValueError(
                f"SRT cue indices must be sequential; expected {expected_index}, found {cue_index}"
            )
        match = _SRT_TIMING.fullmatch(lines[1].strip())
        if match is None:
            raise ValueError(f"SRT cue {expected_index} has an invalid timing line")
        start = _srt_seconds(match.groups()[:4])
        end = _srt_seconds(match.groups()[4:])
        if end <= start:
            raise ValueError(f"SRT cue {expected_index} must have positive duration")
        if start + 0.001 < previous_end:
            raise ValueError(f"SRT cue {expected_index} overlaps the preceding cue")
        if end > media_duration + SRT_END_TOLERANCE_SECONDS:
            raise ValueError(
                f"SRT cue {expected_index} ends after the media duration"
            )
        caption_lines = lines[2:]
        if any(not line.strip() for line in caption_lines):
            raise ValueError(f"SRT cue {expected_index} contains a blank text line")
        if len(caption_lines) > MAX_CAPTION_LINES:
            raise ValueError(
                f"SRT cue {expected_index} exceeds the {MAX_CAPTION_LINES}-line safe band"
            )
        max_lines = max(max_lines, len(caption_lines))
        previous_end = end
    return {
        "caption_cues": len(blocks),
        "caption_max_lines": max_lines,
        "caption_last_end_seconds": previous_end,
    }


def validate_poster(poster: Path) -> dict[str, Any]:
    """Require a decodable release-resolution PNG poster."""
    if not poster.is_file() or poster.stat().st_size == 0:
        raise ValueError(f"poster is missing or empty: {poster}")
    try:
        with Image.open(poster) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"poster is not a decodable image: {poster}") from exc
    if image_format != "PNG":
        raise ValueError(f"poster format must be PNG, found {image_format!r}")
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(
            "poster dimensions must match the release video: "
            f"expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, found {width}x{height}"
        )
    return {
        "poster_width": width,
        "poster_height": height,
        "poster_format": image_format,
    }


def validate_probe(
    probe: dict[str, Any],
    *,
    min_duration: float = MIN_DURATION_SECONDS,
    max_duration: float = MAX_DURATION_SECONDS,
) -> dict[str, Any]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe did not return a stream list")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1 or len(streams) != 2:
        raise ValueError(
            "release video must contain exactly one video and one audio stream "
            f"(found video={len(videos)}, audio={len(audios)}, total={len(streams)})"
        )

    video, audio = videos[0], audios[0]
    if video.get("codec_name") != "h264":
        raise ValueError(f"video codec must be h264, found {video.get('codec_name')!r}")
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(
            f"video resolution must be {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, "
            f"found {width}x{height}"
        )
    if video.get("pix_fmt") != "yuv420p":
        raise ValueError(f"video pixel format must be yuv420p, found {video.get('pix_fmt')!r}")

    if audio.get("codec_name") != "aac":
        raise ValueError(f"audio codec must be aac, found {audio.get('codec_name')!r}")
    if int(audio.get("sample_rate", 0)) != 48_000:
        raise ValueError(
            f"audio sample rate must be 48000 Hz, found {audio.get('sample_rate')!r}"
        )
    if int(audio.get("channels", 0)) != 1:
        raise ValueError(f"audio must be mono, found {audio.get('channels')!r} channels")

    format_data = probe.get("format") or {}
    try:
        duration = float(format_data["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ffprobe did not return a numeric container duration") from exc
    if not min_duration <= duration <= max_duration:
        raise ValueError(
            f"video duration {duration:.3f}s is outside "
            f"{min_duration:.1f}-{max_duration:.1f}s"
        )
    format_names = set(str(format_data.get("format_name", "")).split(","))
    if not {"mov", "mp4"}.intersection(format_names):
        raise ValueError(f"release container must be MP4/MOV, found {sorted(format_names)!r}")
    try:
        video_duration = float(video["duration"])
        audio_duration = float(audio["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ffprobe did not return numeric video/audio stream durations") from exc
    if any(
        abs(stream_duration - duration) > STREAM_DURATION_TOLERANCE_SECONDS
        for stream_duration in (video_duration, audio_duration)
    ):
        raise ValueError(
            "audio and video streams must both span the full container duration: "
            f"container={duration:.3f}, video={video_duration:.3f}, audio={audio_duration:.3f}"
        )

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "video_codec": video["codec_name"],
        "pixel_format": video["pix_fmt"],
        "audio_codec": audio["codec_name"],
        "audio_sample_rate_hz": int(audio["sample_rate"]),
        "audio_channels": int(audio["channels"]),
        "video_stream_duration_seconds": video_duration,
        "audio_stream_duration_seconds": audio_duration,
    }


def validate_release_video(
    video: Path,
    manifest_path: Path,
    captions: Path,
    poster: Path,
    *,
    min_duration: float = MIN_DURATION_SECONDS,
    max_duration: float = MAX_DURATION_SECONDS,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A manifest must name the deliverables it is actually about. The canonical
    # release keeps its exact names, because release day checks those bytes; a
    # named narration variant is held to the same rule against its own names,
    # so a manifest can never describe a file other than the one validated.
    variant = manifest.get("variant")
    if variant is None:
        stem = "coldplate_submission"
    else:
        if not isinstance(variant, str) or not variant.replace("_", "").isalnum():
            raise ValueError("video manifest variant must be an alphanumeric name")
        stem = f"coldplate_submission_{variant}"
    if manifest.get("output") != f"demo/{stem}.mp4":
        raise ValueError(f"video manifest output must name demo/{stem}.mp4")
    if manifest.get("captions") != f"demo/{stem}.en.srt":
        raise ValueError(f"video manifest captions must name demo/{stem}.en.srt")
    if manifest.get("poster") != "demo/poster.png":
        raise ValueError("video manifest poster must name the canonical PNG")
    for path, expected in (
        (video, f"demo/{stem}.mp4"),
        (captions, f"demo/{stem}.en.srt"),
    ):
        if path.name != Path(expected).name:
            raise ValueError(f"validated {path.name} does not match manifest {expected}")
    actual = validate_probe(
        probe_video(video), min_duration=min_duration, max_duration=max_duration
    )
    caption_report = validate_srt(captions, actual["duration_seconds"])
    poster_report = validate_poster(poster)
    audio_report = validate_audio_signal(video)
    expected_scalars = {
        "sha256": sha256_file(video),
        "bytes": video.stat().st_size,
        "captions_sha256": sha256_file(captions),
        "captions_bytes": captions.stat().st_size,
        "poster_sha256": sha256_file(poster),
        "poster_bytes": poster.stat().st_size,
        **poster_report,
        "caption_cues": caption_report["caption_cues"],
        "width": actual["width"],
        "height": actual["height"],
        "video_codec": actual["video_codec"],
        "pixel_format": actual["pixel_format"],
        "audio_codec": actual["audio_codec"],
        "audio_sample_rate_hz": actual["audio_sample_rate_hz"],
        "audio_channels": actual["audio_channels"],
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"video manifest {key!r} does not match the file: "
                f"expected {expected!r}, found {manifest.get(key)!r}"
            )
    try:
        manifest_duration = float(manifest["duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("video manifest has no numeric duration_seconds") from exc
    if abs(manifest_duration - actual["duration_seconds"]) > 0.05:
        raise ValueError(
            "video manifest duration does not match ffprobe: "
            f"{manifest_duration:.3f}s versus {actual['duration_seconds']:.3f}s"
        )
    return {
        **actual,
        **caption_report,
        **poster_report,
        **audio_report,
        "sha256": expected_scalars["sha256"],
        "bytes": expected_scalars["bytes"],
        "captions_sha256": expected_scalars["captions_sha256"],
        "poster_sha256": expected_scalars["poster_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--poster", type=Path, required=True)
    args = parser.parse_args()
    report = validate_release_video(args.video, args.manifest, args.captions, args.poster)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
