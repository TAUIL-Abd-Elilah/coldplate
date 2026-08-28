# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The README's map of its own argument must not contain a dead link.

`check_submission.py` resolves every in-document anchor against GitHub's
heading-slug rule. A slug function that quietly returned the wrong thing would
make that check pass vacuously, so the rule itself is pinned here against
headings this repository actually uses -- including the one that starts with a
Greek letter, and one carrying an underscore that must survive.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checker():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts" / "check_submission.py"
    spec = importlib.util.spec_from_file_location("check_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_submission"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("What we do *not* claim", "what-we-do-not-claim"),
        ("What actually predicts it: one VJP", "what-actually-predicts-it-one-vjp"),
        ("Prior work, and what is actually new here",
         "prior-work-and-what-is-actually-new-here"),
        ("A link to [the paper](PAPER.pdf) here", "a-link-to-the-paper-here"),
        # The underscore is part of a filename, not emphasis, and GitHub keeps it.
        ("Use `coupling_check.py` now", "use-coupling_checkpy-now"),
        ("Does γ generalise past this cold plate? 2,377 random systems say yes "
         "— with one boundary",
         "does-γ-generalise-past-this-cold-plate-2377-random-systems-say-yes--"
         "with-one-boundary"),
    ],
)
def test_slug_matches_github(checker, heading, slug):
    assert checker.github_slug(heading) == slug


def test_every_readme_anchor_resolves(checker):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    slugs = {checker.github_slug(h)
             for h in re.findall(r"^#{1,6}\s+(.*?)\s*$", readme, re.M)}
    anchors = set(re.findall(r"\]\(#([^)]+)\)", readme))
    assert anchors, "the README is expected to link to its own sections"
    assert not anchors - slugs


def test_the_check_can_fail(checker):
    """A checker that cannot reject anything proves nothing."""
    assert checker.github_slug("Some real heading") != "not-a-heading-here"
