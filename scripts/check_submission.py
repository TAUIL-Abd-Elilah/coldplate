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
import re
import subprocess
import sys
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


def main() -> int:  # noqa: C901
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
    sys.exit(main())
