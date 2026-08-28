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


def test_points_file_agrees_with_the_summary(builder):
    """The explorer recomputes from per-trial data; the summary was written by a
    different code path in the same run. If they disagree, one of them is wrong."""
    summary = builder.load("gamma_generalization.json")
    points = builder.load("gamma_generalization_points.json")
    assert points["n"] == summary["trials_usable"] == len(points["rows"])

    gate, danger = builder.SHIPPED_GATE, builder.DANGER
    screened = [err for gamma, err, *_ in points["rows"] if gamma < gate]
    assert len(screened) == summary["safe_bucket"]["n"]
    assert max(screened) == pytest.approx(summary["safe_bucket"]["worst_rel_err"], rel=1e-4)
    under = sum(1 for err in screened if err < 0.05) / len(screened)
    assert under == pytest.approx(summary["safe_bucket"]["frac_under_5pct"])
    # The shipped gate is the one that must not wave through a damaging case.
    assert not [err for err in screened if err > danger]


def test_explorer_shows_a_gate_that_fails(builder):
    """A control that only ever reports good news teaches nothing."""
    points = builder.load("gamma_generalization_points.json")
    worst_gate = max(builder.GATES)
    false_safe = [err for gamma, err, *_ in points["rows"]
                  if gamma < worst_gate and err > builder.DANGER]
    assert false_safe, "expected the loosest offered gate to admit real failures"
    assert "false SAFE" in builder.render()


def test_explorer_refuses_to_plot_outside_its_own_axes(builder, monkeypatch):
    """Axes are fixed; the data is not. A point off the frame must stop the
    build rather than draw over the page."""
    real = builder.load

    def with_an_outlier(name, sub="results"):
        data = real(name, sub)
        if name == "gamma_generalization_points.json":
            data = dict(data)
            data["rows"] = [*data["rows"], [1e-9, 0.5, 0.3, "normal", "linear"]]
        return data

    monkeypatch.setattr(builder, "load", with_an_outlier)
    with pytest.raises(ValueError, match="widen the axis"):
        builder.threshold_explorer()


def test_every_gate_lands_inside_the_plot(builder):
    import math

    for gate in builder.GATES:
        assert builder.XLIM[0] <= math.log10(gate) <= builder.XLIM[1]
    assert builder.SHIPPED_GATE in builder.GATES


def test_the_explorer_is_actually_wired(builder):
    """A CSS selector that names no element fails silently and invisibly.

    The page claims to be interactive. If a rule pointed at a class the markup
    never emits, every gate would look selectable and nothing would move, and
    no other test here would notice.
    """
    page = builder.render()
    style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    body = page[page.index("</style>"):]

    ids = re.findall(r'<input type="radio" name="gate" id="(gate\d)"', body)
    assert len(ids) == len(builder.GATES)
    assert len(re.findall(r'id="gate\d" checked', body)) == 1, (
        "exactly one gate must start selected"
    )

    for gate, container, target in re.findall(
        r"#(gate\d):checked~\.(\w+) \.(\w+)", style
    ):
        assert f'class="{container}"' in body, f"no .{container} for {gate}"
        assert target in body, f"{gate} points at .{target}, which is never emitted"

    for gate in ids:
        assert f'for="{gate}"' in body, f"{gate} has no label to click"
        assert f"label[for={gate}]" in style, f"{gate} never highlights when chosen"
        assert f"#{gate}:focus-visible" in style, f"{gate} has no keyboard focus ring"


def test_page_runs_no_script(builder):
    """The explorer is radio buttons and sibling selectors, deliberately."""
    page = builder.render()
    assert "<script" not in page.lower()
    assert "onclick" not in page.lower()
    assert 'type="radio"' in page


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
