# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_image_manifest import COMPONENTS, parse_manifest  # noqa: E402


def valid_rows() -> list[str]:
    digest = "0123456789abcdef" * 4
    return [
        f"{component}=ghcr.io/tauil-abd-elilah/coldplate-{component}@sha256:{digest}"
        for component in COMPONENTS
    ]


def write_manifest(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "image-digests.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_manifest_requires_exactly_four_digest_pinned_components(tmp_path):
    manifest = write_manifest(tmp_path, valid_rows())
    parsed = parse_manifest(manifest)
    assert tuple(parsed) == COMPONENTS
    assert all("@sha256:" in reference for reference in parsed.values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0]),
        lambda rows: rows.__setitem__(0, rows[0].replace("sha256:", "latest:")),
        lambda rows: rows.__setitem__(0, rows[0][:-1]),
        lambda rows: rows.__setitem__(0, rows[0].replace("stokes_brinkman=", "unknown=")),
    ],
)
def test_manifest_rejects_missing_duplicate_mutable_or_unknown_rows(tmp_path, mutate):
    rows = valid_rows()
    mutate(rows)
    with pytest.raises(ValueError):
        parse_manifest(write_manifest(tmp_path, rows))
