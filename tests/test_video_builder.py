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

from build_demo_video import (  # noqa: E402
    HEIGHT,
    LEAD_SECONDS,
    SECTION_GAP_SECONDS,
    TAIL_SECONDS,
    WIDTH,
    Section,
    _caption_chunks,
    _caption_cues,
    _sentence_timings,
    _timestamp,
    render_slide,
)
import build_demo_video  # noqa: E402
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
                "duration": "240.125",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 1,
                "duration": "240.125",
            },
        ],
        "format": {"duration": "240.125", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    }


def test_srt_timestamp_rounding_and_rollover():
    assert _timestamp(0) == "00:00:00,000"
    assert _timestamp(61.2346) == "00:01:01,235"
    assert _timestamp(3661.001) == "01:01:01,001"


def test_narration_timeline_has_lead_section_gap_and_tail():
    flattened = [(0, "First."), (0, "Second."), (1, "Third.")]
    timings, duration = _sentence_timings(flattened, [1.0, 2.0, 3.0])
    assert timings == [
        (LEAD_SECONDS, LEAD_SECONDS + 1.0, "First."),
        (LEAD_SECONDS + 1.0, LEAD_SECONDS + 3.0, "Second."),
        (
            LEAD_SECONDS + 3.0 + SECTION_GAP_SECONDS,
            LEAD_SECONDS + 6.0 + SECTION_GAP_SECONDS,
            "Third.",
        ),
    ]
    assert duration == pytest.approx(
        LEAD_SECONDS + 6.0 + SECTION_GAP_SECONDS + TAIL_SECONDS
    )


def test_long_caption_is_split_into_safe_nonoverlapping_cues():
    sentence = " ".join(f"word{index}" for index in range(90))
    chunks = _caption_chunks(sentence, width=24, max_lines=3)
    assert len(chunks) > 1
    assert all(1 <= len(chunk.splitlines()) <= 3 for chunk in chunks)
    cues = _caption_cues([(2.0, 12.0, sentence)])
    assert cues[0][0] == 2.0
    assert cues[-1][1] == 12.0
    assert all(left[1] == pytest.approx(right[0]) for left, right in zip(cues, cues[1:]))


def test_video_builder_rejects_noncanonical_output_before_loading_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(build_demo_video.shutil, "which", lambda command: command)
    monkeypatch.setattr(
        build_demo_video,
        "make_story",
        lambda: pytest.fail("evidence should not load before output validation"),
    )
    with pytest.raises(ValueError, match="canonical deliverable"):
        build_demo_video.build(output=tmp_path / "preview.mp4")


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
        (lambda data: data["streams"][1].update(duration="200"), "full container"),
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
    captions.write_text("1\n00:00:00,500 --> 00:00:01,000\nHello\n", encoding="utf-8")
    poster = tmp_path / "poster.png"
    Image.new("RGB", (WIDTH, HEIGHT), "black").save(poster)
    actual = validate_probe(valid_probe())
    manifest = {
        "output": "demo/coldplate_submission.mp4",
        "captions": "demo/coldplate_submission.en.srt",
        "poster": "demo/poster.png",
        **actual,
        "sha256": video_validation.sha256_file(video),
        "bytes": video.stat().st_size,
        "captions_sha256": video_validation.sha256_file(captions),
        "captions_bytes": captions.stat().st_size,
        "poster_sha256": video_validation.sha256_file(poster),
        "poster_bytes": poster.stat().st_size,
        "poster_width": WIDTH,
        "poster_height": HEIGHT,
        "poster_format": "PNG",
        "caption_cues": 1,
    }
    manifest_path = tmp_path / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(video_validation, "probe_video", lambda path: valid_probe())
    monkeypatch.setattr(
        video_validation,
        "validate_audio_signal",
        lambda path: {"mean_volume_db": -16.3, "max_volume_db": -1.4},
    )
    report = video_validation.validate_release_video(
        video, manifest_path, captions, poster
    )
    assert report["bytes"] > 0
    assert report["caption_cues"] == 1

    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        video_validation.validate_release_video(video, manifest_path, captions, poster)


def test_audio_signal_validator_rejects_silent_track(monkeypatch, tmp_path):
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(
        video_validation,
        "probe_audio_levels",
        lambda path: {"mean_volume_db": -90.0, "max_volume_db": -80.0},
    )
    with pytest.raises(ValueError, match="silent"):
        video_validation.validate_audio_signal(video)


def test_release_video_rejects_wrong_size_poster(tmp_path, monkeypatch):
    video = tmp_path / "coldplate_submission.mp4"
    video.write_bytes(b"release-video-bytes")
    captions = tmp_path / "coldplate_submission.en.srt"
    captions.write_text(
        "1\n00:00:00,500 --> 00:00:01,000\nHello\n", encoding="utf-8"
    )
    poster = tmp_path / "poster.png"
    Image.new("RGB", (640, 360), "black").save(poster)
    manifest_path = tmp_path / "video_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(video_validation, "probe_video", lambda path: valid_probe())
    with pytest.raises(ValueError, match="poster dimensions"):
        video_validation.validate_poster(poster)


def test_srt_validator_rejects_overlap_and_more_than_three_lines(tmp_path):
    captions = tmp_path / "bad.srt"
    captions.write_text(
        "1\n00:00:00,500 --> 00:00:02,000\na\nb\nc\nd\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="3-line safe band"):
        video_validation.validate_srt(captions, 3.0)

    captions.write_text(
        "1\n00:00:00,500 --> 00:00:02,000\nfirst\n\n"
        "2\n00:00:01,900 --> 00:00:02,500\nsecond\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlaps"):
        video_validation.validate_srt(captions, 3.0)


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
