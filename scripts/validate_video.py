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
import subprocess
from typing import Any

EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
MIN_DURATION_SECONDS = 180.0
MAX_DURATION_SECONDS = 300.0


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

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "video_codec": video["codec_name"],
        "pixel_format": video["pix_fmt"],
        "audio_codec": audio["codec_name"],
        "audio_sample_rate_hz": int(audio["sample_rate"]),
        "audio_channels": int(audio["channels"]),
    }


def validate_release_video(
    video: Path,
    manifest_path: Path,
    captions: Path,
    *,
    min_duration: float = MIN_DURATION_SECONDS,
    max_duration: float = MAX_DURATION_SECONDS,
) -> dict[str, Any]:
    if not captions.is_file() or captions.stat().st_size == 0:
        raise ValueError(f"captions are missing or empty: {captions}")
    caption_text = captions.read_text(encoding="utf-8")
    if " --> " not in caption_text:
        raise ValueError("captions do not contain an SRT timing cue")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output") != "demo/coldplate_submission.mp4":
        raise ValueError("video manifest output must name the canonical demo MP4")
    if manifest.get("captions") != "demo/coldplate_submission.en.srt":
        raise ValueError("video manifest captions must name the canonical English SRT")
    actual = validate_probe(
        probe_video(video), min_duration=min_duration, max_duration=max_duration
    )
    expected_scalars = {
        "sha256": sha256_file(video),
        "bytes": video.stat().st_size,
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
    return {**actual, "sha256": expected_scalars["sha256"], "bytes": expected_scalars["bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    args = parser.parse_args()
    report = validate_release_video(args.video, args.manifest, args.captions)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
