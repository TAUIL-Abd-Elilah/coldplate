# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
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


def test_release_workflow_pins_actions_and_serializes_manual_phases():
    workflow = (ROOT / ".github" / "workflows" / "release-submission.yml").read_text(
        encoding="utf-8"
    )
    assert "cancel-in-progress: false" in workflow
    assert "permissions: {}" in workflow
    assert workflow.count("name: submission-production") == 2
    assert workflow.count("actions: read") == 2
    assert "2026-08-29 00:00 UTC" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in workflow
    assert "docker/login-action@dbcb813823bdd20940b903addbd779551569679f" in workflow
    assert "actions/checkout@v" not in workflow
    assert "actions/setup-python@v" not in workflow
    assert "docker/login-action@v" not in workflow
    assert "python -m pip install -r requirements-video.txt" in workflow
    assert "validate_evidence_provenance.py --verify-github" in workflow
    assert workflow.count('gh api "repos/$GITHUB_REPOSITORY" --jq .visibility') == 2


def test_release_workflow_never_interpolates_confirmation_into_shell_source():
    workflow = (ROOT / ".github" / "workflows" / "release-submission.yml").read_text(
        encoding="utf-8"
    )
    assert "test '${{ inputs.confirmation }}'" not in workflow
    assert workflow.count("CONFIRMATION: ${{ inputs.confirmation }}") == 2
    assert 'os.environ["CONFIRMATION"] != "PREPARE"' in workflow
    assert 'os.environ["CONFIRMATION"] != "PUBLISH"' in workflow


def test_publish_uses_checksummed_provenance_and_exact_prepared_checkout():
    workflow = (ROOT / ".github" / "workflows" / "release-submission.yml").read_text(
        encoding="utf-8"
    )
    assert "release-provenance.json" in workflow
    assert '"prepared_sha": sha' in workflow
    assert "sha256sum -c SHA256SUMS" in workflow
    assert "ref: ${{ steps.provenance.outputs.prepared_sha }}" in workflow
    assert 'test "$(git rev-parse HEAD)" = "$PREPARED_SHA"' in workflow
    assert 'state.get("isDraft")' in workflow
    assert "python scripts/validate_video.py" in workflow


def test_release_cli_always_names_the_repository_explicitly():
    """Pre-checkout release steps cannot rely on a local git repository."""
    workflow = (ROOT / ".github" / "workflows" / "release-submission.yml").read_text(
        encoding="utf-8"
    )
    logical_lines = workflow.replace("\\\n", " ").splitlines()
    release_commands = [line for line in logical_lines if "gh release " in line]
    assert release_commands
    assert all('--repo "$GITHUB_REPOSITORY"' in command for command in release_commands)


def test_every_github_action_is_pinned_to_a_full_commit_sha():
    workflows = ROOT / ".github" / "workflows"
    uses = []
    for path in workflows.glob("*.yml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- uses:"):
                uses.append((path.name, stripped.split("uses:", 1)[1].strip().split()[0]))
    assert uses
    bad = [(name, value) for name, value in uses
           if "@" not in value
           or re.fullmatch(r"[0-9a-f]{40}", value.rsplit("@", 1)[1]) is None]
    assert not bad


def test_enzyme_plugin_is_vendored_licensed_and_byte_pinned():
    dockerfile = (
        ROOT / "tesseracts" / "thermal_fortran" / "toolchain" / "Dockerfile"
    ).read_text(encoding="utf-8")
    vendor = ROOT / "tesseracts" / "thermal_fortran" / "toolchain" / "vendor"
    plugin = vendor / "LLVMEnzyme-19.so"
    licence = vendor / "LICENSE.Enzyme.txt"
    expected = "5b43014ab23fdf212b5c0852e5ae1d2e9d3062bf0aa2323bbbf63b33369ef031"
    assert plugin.stat().st_size == 8_050_632
    assert hashlib.sha256(plugin.read_bytes()).hexdigest() == expected
    assert hashlib.sha256(licence.read_bytes()).hexdigest() == (
        "f2db94d30c9657f2556732f3e80973d49fc4d093eede0a54ffda88152296f695"
    )
    assert "Apache License v2.0 with LLVM Exceptions" in licence.read_text(
        encoding="utf-8"
    )
    assert "COPY vendor/LLVMEnzyme-19.so" in dockerfile
    assert "COPY vendor/LICENSE.Enzyme.txt" in dockerfile
    assert expected in dockerfile
    assert "releases/download/nightly" not in dockerfile


def test_evidence_workflow_stages_only_fresh_outputs_and_fails_invalid_physics():
    workflow = (ROOT / ".github" / "workflows" / "evidence-v2.yml").read_text(
        encoding="utf-8"
    )
    assert '$RUNNER_TEMP/extended-evidence' in workflow
    assert "path: ${{ runner.temp }}/extended-evidence/" in workflow
    assert "physical.get(\"evidence_valid\") is True" in workflow
    assert "all_solves_converged\") is True" in workflow
    assert "path: |\n            orchestrator/results/" not in workflow
    assert "- all" not in workflow


def test_paper_build_fixes_font_timestamp_for_reproducible_pdf_bytes():
    script = (ROOT / "scripts" / "build_paper.sh").read_text(encoding="utf-8")
    assert 'SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1785715200}"' in script
    assert 'PYTHONHASHSEED="${PYTHONHASHSEED:-0}"' in script


def test_release_publishes_both_narrations():
    """The rights-clean render must not be able to fall out of a release.

    It exists precisely so the submission never depends on audio whose
    redistribution rights we could not confirm. A release that shipped only
    the canonical MP4 would quietly undo that, so every stage that names the
    canonical deliverable has to name this one too.
    """
    workflow = (ROOT / ".github" / "workflows" / "release-submission.yml").read_text(
        encoding="utf-8"
    )
    variant = {
        "mp4": "coldplate_submission_local_voice.mp4",
        "srt": "coldplate_submission_local_voice.en.srt",
        "manifest": "video_manifest_local_voice.json",
    }
    canonical = {
        "mp4": "coldplate_submission.mp4",
        "srt": "coldplate_submission.en.srt",
        "manifest": "video_manifest.json",
    }
    for key, name in variant.items():
        # The canonical name is a substring of nothing here, but the variant
        # name contains no canonical name, so plain counting is unambiguous.
        assert name in workflow, f"the release workflow never mentions {name}"
        # Wherever the canonical deliverable is enumerated, so is this one.
        assert workflow.count(name) >= 4, (
            f"{name} appears {workflow.count(name)} times; the canonical "
            f"{canonical[key]} is enumerated in the preflight asset sets, the "
            "required-files list, the asset copy, the checksums and the "
            "publish-phase verification"
        )
    assert workflow.count("--video demo/coldplate_submission_local_voice.mp4") == 1
    assert workflow.count(
        '--video "$VERIFY_DIR/coldplate_submission_local_voice.mp4"'
    ) == 1


def test_third_party_notices_state_the_narration_position():
    """The narration licence position is load-bearing; it must stay explicit."""
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for phrase in (
        # the admission about the canonical render
        "no written confirmation",
        # why the local render is clean
        "public domain",
        # and the licence of the tool that produced it, stated rather than glossed
        "GPL-3.0-or-later",
        "not** redistribute it",
    ):
        assert phrase in notices, f"THIRD_PARTY_NOTICES lost {phrase!r}"
    # Both renders must be named, so a reader can find the clean one.
    assert "demo/coldplate_submission.mp4" in notices
    assert "demo/coldplate_submission_local_voice.mp4" in notices
