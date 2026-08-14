# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_demo_video import HEIGHT, WIDTH, Section, _timestamp, render_slide  # noqa: E402


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
