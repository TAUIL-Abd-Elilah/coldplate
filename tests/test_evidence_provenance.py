# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_evidence_provenance import tree_digest, validate_manifest  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    orchestrator = tmp_path / "orchestrator"
    orchestrator.mkdir()
    (orchestrator / "interpret_showdown.py").write_text("# generator\n", encoding="utf-8")
    workflow = tmp_path / ".github" / "workflows" / "evidence-v2.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: evidence\n", encoding="utf-8")
    results = orchestrator / "results"
    attempts = results / "intervention_robustness_matrix_48_attempts"
    attempts.mkdir(parents=True)
    matrix = results / "intervention_robustness_matrix_48.json"
    showdown = results / "strong_coupling_showdown.json"
    interpretation = results / "strong_coupling_showdown_interpretation.json"
    cavity = results / "de_vahl_davis.json"
    dimensional = results / "dimensional_coldplate.json"
    matrix.write_text('{"complete": true}\n', encoding="utf-8")
    showdown.write_text('{"complete": false}\n', encoding="utf-8")
    interpretation.write_text('{"verdict": null}\n', encoding="utf-8")
    cavity.write_text('[{"valid": true}]\n', encoding="utf-8")
    dimensional.write_text('{"evidence_valid": false}\n', encoding="utf-8")
    for index in range(48):
        (attempts / f"attempt-{index:02d}.json").write_text(
            json.dumps({"index": index}) + "\n", encoding="utf-8"
        )
    tree_hash, count = tree_digest(attempts)

    def artifact_record(
        name: str,
        *,
        source: str,
        prefix: str,
        result_status: str,
        files: dict[str, str],
        conclusion: str = "success",
        trees: list[dict] | None = None,
    ) -> dict:
        return {
            "name": name,
            "kind": "github_actions_artifact",
            "result_status": result_status,
            "run_id": 123,
            "run_url": "https://github.com/TAUIL-Abd-Elilah/coldplate/actions/runs/123",
            "source_sha": source,
            "workflow_file": ".github/workflows/evidence-v2.yml",
            "conclusion": conclusion,
            "artifact": {
                "id": 456,
                "name": prefix + source,
                "digest": "sha256:" + "d" * 64,
                "size_in_bytes": 1234,
            },
            "files": files,
            "trees": trees or [],
        }

    payload = {
        "schema": "coldplate-evidence-provenance-v1",
        "repository": "TAUIL-Abd-Elilah/coldplate",
        "records": [
            artifact_record(
                "robustness_matrix_48",
                source="a" * 40,
                prefix="extended-evidence-robustness-",
                result_status="complete",
                files={
                    "orchestrator/results/intervention_robustness_matrix_48.json": _sha(matrix)
                },
                trees=[{
                    "path": "orchestrator/results/intervention_robustness_matrix_48_attempts",
                    "pattern": "*.json",
                    "file_count": count,
                    "sha256": tree_hash,
                }],
            ),
            artifact_record(
                "strong_coupling_showdown",
                source="b" * 40,
                prefix="extended-evidence-showdown-",
                result_status="incomplete_retained",
                conclusion="expected_failure",
                files={"orchestrator/results/strong_coupling_showdown.json": _sha(showdown)},
            ),
            artifact_record(
                "physics_bundle",
                source="c" * 40,
                prefix="extended-evidence-physics-",
                result_status="cavity_valid_dimensional_invalid",
                files={
                    "orchestrator/results/de_vahl_davis.json": _sha(cavity),
                    "orchestrator/results/dimensional_coldplate.json": _sha(dimensional),
                },
            ),
            {
                "name": "showdown_interpretation",
                "kind": "derived",
                "result_status": "withholds_endpoint_verdict",
                "generator": "orchestrator/interpret_showdown.py",
                "inputs": {
                    "orchestrator/results/strong_coupling_showdown.json": _sha(showdown)
                },
                "files": {
                    "orchestrator/results/strong_coupling_showdown_interpretation.json": _sha(interpretation)
                },
                "trees": [],
            },
        ],
    }
    manifest = results / "EVIDENCE_PROVENANCE.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_valid_manifest_binds_artifact_files_tree_and_derived_input(tmp_path):
    report = validate_manifest(_fixture(tmp_path), root=tmp_path)
    assert report == {
        "records": 4,
        "names": [
            "physics_bundle",
            "robustness_matrix_48",
            "showdown_interpretation",
            "strong_coupling_showdown",
        ],
    }


def test_manifest_rejects_mutated_evidence(tmp_path):
    manifest = _fixture(tmp_path)
    (tmp_path / "orchestrator" / "results" / "strong_coupling_showdown.json").write_text(
        '{"complete": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(manifest, root=tmp_path)


def test_manifest_rejects_path_escape(tmp_path):
    manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][1]["files"] = {"../outside.json": "0" * 64}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe repository-relative path"):
        validate_manifest(manifest, root=tmp_path)


def test_manifest_requires_every_canonical_record(tmp_path):
    manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"] = payload["records"][:-1]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_manifest(manifest, root=tmp_path)
