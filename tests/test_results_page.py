# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The published results page must stay a pure function of the measurements.

`docs/index.html` quotes roughly forty numbers. Every one of them is read out
of a JSON file in `orchestrator/results/` at build time, so the only way the
page can disagree with the evidence is if someone edits the HTML by hand or
regenerates the evidence without regenerating the page. Both are build
failures here rather than something a reader is expected to catch.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"


@pytest.fixture(scope="module")
def builder():
    path = ROOT / "scripts" / "build_results_page.py"
    spec = importlib.util.spec_from_file_location("build_results_page", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_results_page"] = module
    spec.loader.exec_module(module)
    return module


def test_committed_page_matches_the_measurements(builder):
    assert PAGE.exists(), "docs/index.html is missing; run scripts/build_results_page.py"
    assert PAGE.read_text(encoding="utf-8") == builder.render(), (
        "docs/index.html is stale -- rerun scripts/build_results_page.py"
    )


def test_page_is_deterministic(builder):
    assert builder.render() == builder.render()


def test_page_is_self_contained(builder):
    """No CDN, no tracker, no font fetch: it has to open from a clone, offline.

    Ordinary anchors may of course point at the web; what must not happen is the
    page *loading* anything over the network to render.
    """
    page = builder.render()
    remote = re.findall(r'src\s*=\s*"(?:https?:)?//[^"]+"', page)
    remote += re.findall(r"<link[^>]+href\s*=\s*\"(?:https?:)?//[^\"]+\"", page)
    remote += re.findall(r"@import[^;]+", page)
    assert not remote, f"page pulls remote subresources: {remote}"
    assert "<script" not in page.lower()


def test_page_links_resolve_from_the_docs_directory(builder):
    page = builder.render()
    targets = {
        target
        for target in re.findall(r'(?:src|href)\s*=\s*"([^"#]+)"', page)
        if not target.startswith(("http", "#", "mailto:"))
    }
    assert targets, "expected the page to reference repository assets"
    missing = sorted(str(t) for t in targets if not (PAGE.parent / t).resolve().exists())
    assert not missing, f"broken relative links on the results page: {missing}"


def test_page_keeps_the_negative_results(builder):
    """The section a reader would most want removed is the one that must stay."""
    page = builder.render()
    for phrase in (
        "not evaluable",
        "no verdict",
        "failed audit",
        "worthless as an attribution",
        "where it stops working",
    ):
        assert phrase in page, f"the results page dropped {phrase!r}"
