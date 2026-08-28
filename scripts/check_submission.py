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
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from validate_evidence_provenance import validate_manifest
from validate_video import validate_release_video

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


def github_slug(heading: str) -> str:
    """GitHub's anchor rule: drop inline markup, lowercase, punctuation out,
    spaces to hyphens. Unicode letters survive, which is why the gamma heading
    anchors as itself."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"", heading)
    text = re.sub(r"`([^`]*)`", r"", text)
    text = re.sub(r"[*_]+", "", text)
    text = text.lower().replace(" ", "-")
    return "".join(c for c in text if c.isalnum() or c in "-_")


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
    evidence_provenance = ROOT / "orchestrator" / "results" / "EVIDENCE_PROVENANCE.json"
    if evidence_provenance.is_file():
        try:
            report = validate_manifest(evidence_provenance)
            check(
                "extended evidence is hash-bound to workflow artifacts",
                True,
                f"{report['records']} provenance records",
            )
        except Exception as exc:  # noqa: BLE001 - readiness check reports all failures
            check("extended evidence is hash-bound to workflow artifacts", False, str(exc))
    elif strict_public:
        check("extended evidence is hash-bound to workflow artifacts", False, "manifest missing")
    else:
        warn("extended evidence provenance manifest is not generated yet")
    video = ROOT / "demo" / "coldplate_submission.mp4"
    captions = ROOT / "demo" / "coldplate_submission.en.srt"
    poster = ROOT / "demo" / "poster.png"
    video_manifest = ROOT / "demo" / "video_manifest.json"
    media_deliverables = (
        ("rendered demo video present", video),
        ("English demo captions present", captions),
        ("demo poster present", poster),
        ("video manifest present", video_manifest),
    )
    media_present = [path.is_file() and path.stat().st_size > 0
                     for _, path in media_deliverables]
    if strict_public or any(media_present):
        for (label, _), present in zip(media_deliverables, media_present):
            check(label, present)
    else:
        warn(
            "rendered video deliverables are not generated yet",
            "allowed only during private development; --strict-public makes them mandatory",
        )
    if all(media_present) and shutil.which("ffprobe") is None and not strict_public:
        # Reviewers are not obliged to have ffmpeg installed. Missing ffprobe
        # means the stream checks cannot run here, not that the deliverables
        # are wrong -- warn instead of failing. --strict-public still demands
        # a real verification before anything is published.
        warn(
            "ffprobe not on PATH, so video stream verification was skipped",
            "install ffmpeg to verify locally; CI and --strict-public require it",
        )
    elif all(media_present):
        try:
            media = validate_release_video(video, video_manifest, captions, poster)
            check(
                "video has verified 1080p H.264 + AAC streams and is under five minutes",
                True,
                f"{media['duration_seconds']:.3f}s, {media['sha256'][:12]}...",
            )
        except Exception as exc:  # noqa: BLE001 - turn tool/format errors into readiness failures
            check(
                "video has verified 1080p H.264 + AAC streams and is under five minutes",
                False,
                str(exc),
            )
    # The browsable results page. Its numbers are generated from the stored
    # measurements, so a stale page is a defect rather than a cosmetic drift.
    page = ROOT / "docs" / "index.html"
    check("browsable results page present", page.is_file() and page.stat().st_size > 0)
    if page.is_file():
        stale = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_results_page.py"), "--check"],
            cwd=ROOT, capture_output=True, text=True,
        )
        check("results page still matches the committed measurements",
              stale.returncode == 0, stale.stdout.strip().splitlines()[-1] if stale.stdout else "")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("README names the track", "Multi-physics" in readme or "multi-physics" in readme)
    check("README justifies why Tesseract is needed",
          "Why this needs Tesseract" in readme)
    check("README embeds the hero and architecture visuals",
          "fig1_optimisation.gif" in readme and "fig5_architecture.png" in readme)

    # Direct invocation is the documented Linux golden path. Git on Windows
    # can hide missing execute bits (core.filemode=false), so inspect the index
    # rather than the local filesystem permissions.
    try:
        modes = subprocess.run(
            ["git", "ls-files", "--stage", "scripts/*.sh"], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.splitlines()
        bad_modes = [line.split(maxsplit=1)[1] for line in modes
                     if line and not line.startswith("100755 ")]
        check("shell scripts are executable in fresh Linux clones", not bad_modes,
              ", ".join(bad_modes[:4]))
    except Exception as exc:  # noqa: BLE001
        warn("could not verify shell-script execute bits", str(exc))
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

    # The README opens with a map of its own argument. A dead anchor there is
    # the first thing a reviewer would click, so resolve every one of them
    # against GitHub's heading-slug rules rather than trusting them.
    headings = re.findall(r"^#{1,6}\s+(.*?)\s*$", readme, re.M)
    slugs = {github_slug(h) for h in headings}
    anchors = set(re.findall(r"\]\(#([^)]+)\)", readme))
    dead = sorted(anchors - slugs)
    check(f"{len(anchors)} README section links resolve", not dead, str(dead))

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

    # A block is copied and pasted as a unit, so every line after the first
    # starts in the directory the previous line left. Two relative `cd`s into
    # the same directory therefore cannot both succeed -- the second fails with
    # "No such file or directory". Checking only that the *targets* exist misses
    # this entirely, which is exactly how it shipped: both commands named real
    # scripts and the check reported them runnable.
    # Counting `cd`s is not enough: `cd coldplate` then `cd orchestrator` is
    # perfectly valid. Track a virtual working directory instead and flag only
    # a cd whose target does not exist from where the previous line left us.
    broken = []
    for block in cmds:
        cwd = ROOT
        for line in block.strip().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cd = re.match(r"cd ([\w./-]+)", s)
            if not cd:
                continue
            target = cd.group(1)
            if target.startswith("/"):
                break  # absolute path; nothing repo-relative left to verify
            nxt = (cwd / target).resolve()
            if not nxt.is_dir():
                broken.append(f"`{s[:58]}` after cwd={cwd.relative_to(ROOT) or '.'}")
                break
            cwd = nxt
    check("every bash block's directory changes work when pasted as a unit",
          not broken, str(broken))

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
