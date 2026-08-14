# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_demo_video import HEIGHT, WIDTH, Section, _timestamp, render_slide  # noqa: E402
import validate_video as video_validation  # noqa: E402
from validate_video import probe_video, validate_probe  # noqa: E402


def valid_probe() -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": WIDTH,
                "height": HEIGHT,
                "pix_fmt": "yuv420p",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 1,
            },
        ],
        "format": {"duration": "240.125", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    }


def test_srt_timestamp_rounding_and_rollover():
    assert _timestamp(0) == "00:00:00,000"
    assert _timestamp(61.2346) == "00:01:01,235"
    assert _timestamp(3661.001) == "01:01:01,001"


def test_slide_renderer_produces_release_resolution(tmp_path):
    asset = tmp_path / "asset.png"
    Image.new("RGB", (800, 300), "white").save(asset)
    section = Section(
        title="A real fixed point",
        kicker="TEST EVIDENCE",
        claim="SAME START · SAME ACTION · TRUE FORWARD SOLVE",
        asset=asset,
        sentences=("One sentence.",),
    )
    output = tmp_path / "slide.png"
    render_slide(section, 1, 1, output)
    with Image.open(output) as rendered:
        assert rendered.size == (WIDTH, HEIGHT)
        assert rendered.mode == "RGB"


def test_probe_validator_checks_real_stream_properties():
    report = validate_probe(valid_probe())
    assert report["duration_seconds"] == 240.125
    assert report["video_codec"] == "h264"
    assert report["audio_codec"] == "aac"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data["streams"].pop(), "one video and one audio"),
        (lambda data: data["streams"][0].update(width=1280), "1920x1080"),
        (lambda data: data["streams"][1].update(sample_rate="44100"), "48000"),
        (lambda data: data["format"].update(duration="300.001"), "outside"),
    ],
)
def test_probe_validator_rejects_broken_delivery(mutate, message):
    data = valid_probe()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        validate_probe(data)


def test_manifest_is_checked_against_probed_file(tmp_path, monkeypatch):
    video = tmp_path / "coldplate_submission.mp4"
    video.write_bytes(b"release-video-bytes")
    captions = tmp_path / "coldplate_submission.en.srt"
    captions.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    actual = validate_probe(valid_probe())
    manifest = {
        "output": "demo/coldplate_submission.mp4",
        "captions": "demo/coldplate_submission.en.srt",
        **actual,
        "sha256": video_validation.sha256_file(video),
        "bytes": video.stat().st_size,
    }
    manifest_path = tmp_path / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(video_validation, "probe_video", lambda path: valid_probe())
    assert video_validation.validate_release_video(video, manifest_path, captions)["bytes"] > 0

    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        video_validation.validate_release_video(video, manifest_path, captions)


def test_ffprobe_round_trip_on_muxed_sample(tmp_path):
    output = tmp_path / "sample.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30",
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                "-t", "0.25", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
                "-ac", "1", str(output),
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg with libx264 is unavailable")
    report = validate_probe(probe_video(output), min_duration=0.1, max_duration=1.0)
    assert (report["width"], report["height"]) == (WIDTH, HEIGHT)
