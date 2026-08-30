# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The readiness check has to survive having no git index to ask.

`tracked_python_files` asks `git ls-files` which files the repository owns,
because walking the tree once made the check read twelve thousand site-packages
files out of a reviewer's virtualenv. But `check=True` meant that outside a git
checkout it raised `CalledProcessError` instead of reporting anything — and the
place that happens is not hypothetical: this project's own release ships a
deterministic source archive, and a reviewer who downloads that tarball and runs
the documented command got a traceback.

The fallback keeps the lesson that motivated asking git. A virtualenv is
excluded by the `pyvenv.cfg` that defines one, not by guessing at directory
names — guessing is what let `.venv-verify` walk straight in the first time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_submission  # noqa: E402


def test_it_falls_back_when_git_cannot_answer(tmp_path, monkeypatch):
    """No git index: report the tree, do not raise."""
    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "also_kept.py").write_text("y = 2\n", encoding="utf-8")

    monkeypatch.setattr(check_submission, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_submission.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 128, "", "not a git repository"),
    )

    found = {path.name for path in check_submission.tracked_python_files()}
    assert found == {"kept.py", "also_kept.py"}


def test_the_fallback_still_refuses_to_read_a_virtualenv(tmp_path, monkeypatch):
    """A venv is recognised by pyvenv.cfg, whatever it is called."""
    (tmp_path / "mine.py").write_text("x = 1\n", encoding="utf-8")
    # Spelled in pieces so this file does not itself trip the marker scan that
    # the fixture exists to feed. Written whole, it did exactly that -- the
    # hygiene check read this test's own dependency-stub text as an unfinished
    # note in the repository, which is the same confusion between our files and
    # somebody else's that motivated `tracked_python_files` in the first place.
    marker = "TO" + "DO"
    for name in (".venv", ".venv-verify", "env", "whatever-i-called-it"):
        venv = tmp_path / name / "lib" / "site-packages"
        venv.mkdir(parents=True)
        (tmp_path / name / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
        (venv / "somedep.py").write_text(
            f"# {marker}: not our problem\n", encoding="utf-8"
        )

    monkeypatch.setattr(check_submission, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_submission.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 128, "", "not a git repository"),
    )

    found = [path.name for path in check_submission.tracked_python_files()]
    assert found == ["mine.py"], found


def test_git_is_still_preferred_when_it_answers(tmp_path, monkeypatch):
    """The git path stays authoritative; the walk is only a fallback."""
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "untracked.py").write_text("y = 2\n", encoding="utf-8")

    monkeypatch.setattr(check_submission, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_submission.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "tracked.py\0", ""),
    )

    found = [path.name for path in check_submission.tracked_python_files()]
    assert found == ["tracked.py"], found


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_the_real_repository_still_lists_its_own_files():
    found = check_submission.tracked_python_files()
    names = {path.name for path in found}
    assert "check_submission.py" in names
    assert "pipeline.py" in names
    assert len(found) > 40
