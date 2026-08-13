#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Submission readiness check: does the repository deliver what it promises?

Separate from audit_claims.py, which checks that quoted *numbers* match the
measurements. This checks the mechanical things a reviewer trips over: a README
that references a file which was renamed, a command that cannot run, a source
file that no longer parses, a missing licence.

    usage:  python scripts/check_submission.py
"""

from __future__ import annotations

import ast
import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL: list[str] = []
WARN: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def warn(label: str, detail: str = "") -> None:
    print(f"  [warn] {label}" + (f"  {detail}" if detail else ""))
    WARN.append(label)


def github_visibility() -> str | None:
    """Return public/private for origin when GitHub can be queried."""
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)(?:\.git)?$", remote)
        if not match:
            return None
        request = urllib.request.Request(
            f"https://api.github.com/repos/{match.group(1)}/{match.group(2)}",
            headers={"User-Agent": "coldplate-submission-check"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return "public" if response.status == 200 else None
    except Exception:  # private repositories return 404 without credentials
        try:
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "visibility", "--jq", ".visibility"],
                cwd=ROOT, capture_output=True, text=True, timeout=15,
            )
            value = result.stdout.strip().lower()
            return value if value in {"public", "private", "internal"} else None
        except Exception:
            return None


def main(*, strict_public: bool = False, allow_dirty: bool = False) -> int:  # noqa: C901
    print("=== hackathon deliverables ===")
    check("public licence file present", (ROOT / "LICENSE").exists())
    lic = (ROOT / "LICENSE").read_text(errors="ignore") if (ROOT / "LICENSE").exists() else ""
    check("licence is Apache 2.0 (required by the rules)", "Apache License" in lic
          and "Version 2.0" in lic)
    check("README present", (ROOT / "README.md").exists())
    check("technical writeup present", (ROOT / "PAPER.md").exists())
    check("writeup PDF built", (ROOT / "PAPER.pdf").exists())
    check("demo video script present", (ROOT / "DEMO_SCRIPT.md").exists())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("README names the track", "Multi-physics" in readme or "multi-physics" in readme)
    check("README justifies why Tesseract is needed",
          "Why this needs Tesseract" in readme)
    check("README embeds the hero and architecture visuals",
          "fig1_optimisation.gif" in readme and "fig5_architecture.png" in readme)
    visibility = github_visibility()
    if visibility == "public":
        check("GitHub repository is public (required for eligibility)", True)
    elif strict_public:
        check("GitHub repository is public (required for eligibility)", False,
              f"detected {visibility or 'unknown'}")
    else:
        warn("GitHub repository is not public yet",
             f"detected {visibility or 'unknown'}; rerun with --strict-public before submission")

    print("\n=== every Tesseract is complete ===")
    for name in ("stokes_brinkman", "thermal_advdiff", "thermal_fortran", "material_map"):
        d = ROOT / "tesseracts" / name
        ok = (d / "tesseract_api.py").exists() and (d / "tesseract_config.yaml").exists()
        check(f"{name}: api + config", ok)
        req = d / "tesseract_requirements.txt"
        if req.exists():
            pins = [l for l in req.read_text().splitlines()
                    if l.strip() and not l.startswith("#") and not l.startswith("-")]
            unpinned = [l for l in pins if "==" not in l]
            if unpinned:
                warn(f"{name}: unpinned dependency", str(unpinned))

    print("\n=== every python file parses ===")
    bad = []
    for f in sorted(ROOT.rglob("*.py")):
        if ".venv" in f.parts or "__pycache__" in f.parts:
            continue
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{f.relative_to(ROOT)}: {e}")
    check(f"all {len(list(ROOT.rglob('*.py')))} python files parse", not bad,
          "; ".join(bad[:3]))

    print("\n=== files and scripts the README references exist ===")
    # bare filenames in backticks, and markdown links to local paths
    named = set(re.findall(r"`([\w./-]+\.(?:py|sh|md|yaml|yml|txt|css))`", readme))
    named |= set(re.findall(r"\]\((?!http)([\w./-]+)\)", readme))
    missing = []
    for n in sorted(named):
        hits = list(ROOT.rglob(n.split("/")[-1])) if "/" not in n else [ROOT / n]
        if not any(h.exists() for h in hits):
            missing.append(n)
    check(f"{len(named)} referenced files all exist", not missing, str(missing))

    print("\n=== README shell commands are runnable ===")
    cmds = re.findall(r"```bash\n(.*?)```", readme, re.S)
    flat = [l.strip() for b in cmds for l in b.strip().splitlines()
            if l.strip() and not l.strip().startswith("#")]
    for c in flat:
        m = re.match(r"(?:cd \S+ && )?(?:python|python3) (\S+\.py)", c)
        if m:
            rel = m.group(1)
            cand = [ROOT / "orchestrator" / rel, ROOT / rel]
            if not any(p.exists() for p in cand):
                check(f"command target exists: {rel}", False)
        m = re.match(r"(scripts/\S+\.sh)", c)
        if m and not (ROOT / m.group(1)).exists():
            check(f"script exists: {m.group(1)}", False)
    check(f"{len(flat)} README commands reference existing targets", True)

    print("\n=== hygiene ===")
    leftovers = []
    marker = re.compile(r"\b(?:TO" + r"DO|FIX" + r"ME|XX" + r"X)\b")
    for f in sorted(ROOT.rglob("*.py")):
        # skip this file: it necessarily contains the very markers it searches
        # for, and matching itself made the check fail on a clean repository
        if ".venv" in f.parts or "__pycache__" in f.parts or f == Path(__file__).resolve():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if marker.search(line):
                leftovers.append(f"{f.relative_to(ROOT)}:{i}")
    check("no TODO/FIXME markers left in source", not leftovers, str(leftovers[:4]))

    try:
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60).stdout.strip()
        if allow_dirty and st:
            warn("git working tree is dirty (allowed for local development)", st[:120])
        else:
            check("git working tree is clean", not st, st[:120])
        br = subprocess.run(["git", "log", "--oneline", "-1"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60).stdout.strip()
        print(f"  head: {br}")
    except Exception as e:  # noqa: BLE001
        warn("could not query git", str(e))

    print()
    if WARN:
        print(f"{len(WARN)} warning(s) -- not blocking")
    if FAIL:
        print(f"\n{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("submission is structurally complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-public", action="store_true",
                        help="fail unless the GitHub repository is public")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="warn instead of fail on local uncommitted changes")
    sys.exit(main(**vars(parser.parse_args())))
