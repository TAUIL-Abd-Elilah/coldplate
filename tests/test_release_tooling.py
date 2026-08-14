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
