#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the offline manifest binding evidence files to Actions artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "orchestrator" / "results" / "EVIDENCE_PROVENANCE.json"
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPOSITORY = "TAUIL-Abd-Elilah/coldplate"
RESULTS_PREFIX = PurePosixPath("orchestrator/results")
EXPECTED_RECORDS = {
    "robustness_matrix_48": {
        "kind": "github_actions_artifact",
        "result_status": "complete",
        "artifact_prefix": "extended-evidence-robustness-",
        "files": {"orchestrator/results/intervention_robustness_matrix_48.json"},
        "trees": {"orchestrator/results/intervention_robustness_matrix_48_attempts": 48},
    },
    "strong_coupling_showdown": {
        "kind": "github_actions_artifact",
        "result_status": "incomplete_retained",
        "artifact_prefix": "extended-evidence-showdown-",
        "files": {"orchestrator/results/strong_coupling_showdown.json"},
        "trees": {},
    },
    "physics_bundle": {
        "kind": "github_actions_artifact",
        "result_status": "cavity_valid_dimensional_invalid",
        "artifact_prefix": "extended-evidence-physics-",
        "files": {
            "orchestrator/results/de_vahl_davis.json",
            "orchestrator/results/dimensional_coldplate.json",
        },
        "trees": {},
    },
    "showdown_interpretation": {
        "kind": "derived",
        "result_status": "withholds_endpoint_verdict",
        "generator": "orchestrator/interpret_showdown.py",
        "inputs": {"orchestrator/results/strong_coupling_showdown.json"},
        "files": {"orchestrator/results/strong_coupling_showdown_interpretation.json"},
        "trees": {},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(root: Path, value: str) -> Path:
    logical = PurePosixPath(value)
    if logical.is_absolute() or not logical.parts or ".." in logical.parts:
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    candidate = (root / Path(*logical.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {value!r}") from exc
    return candidate


def tree_digest(directory: Path, pattern: str = "*.json") -> tuple[str, int]:
    """Hash a sorted list of relative names and individual file digests."""
    files = sorted(path for path in directory.rglob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _validate_files(root: Path, files: dict[str, Any], label: str) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} must bind at least one file")
    for logical, expected in files.items():
        if not isinstance(logical, str) or not isinstance(expected, str):
            raise ValueError(f"{label} file bindings must be string pairs")
        path = _repo_path(root, logical)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} file is missing or empty: {logical}")
        normalized = expected.removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError(f"{label} has an invalid SHA-256 for {logical}")
        actual = sha256_file(path)
        if actual != normalized:
            raise ValueError(
                f"{label} hash mismatch for {logical}: expected {normalized}, found {actual}"
            )


def _validate_trees(root: Path, trees: list[dict[str, Any]], label: str) -> None:
    if not isinstance(trees, list):
        raise ValueError(f"{label} trees must be a list")
    for tree in trees:
        if not isinstance(tree, dict):
            raise ValueError(f"{label} tree record must be an object")
        logical = tree.get("path")
        pattern = tree.get("pattern", "*.json")
        expected_count = tree.get("file_count")
        expected_hash = tree.get("sha256")
        if (
            not isinstance(logical, str)
            or not isinstance(pattern, str)
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 1
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ValueError(f"{label} has an invalid tree record")
        directory = _repo_path(root, logical)
        if not directory.is_dir():
            raise ValueError(f"{label} tree directory is missing: {logical}")
        actual_hash, actual_count = tree_digest(directory, pattern)
        if (actual_count, actual_hash) != (expected_count, expected_hash):
            raise ValueError(
                f"{label} tree mismatch for {logical}: expected "
                f"{expected_count}/{expected_hash}, found {actual_count}/{actual_hash}"
            )


def _validate_expected_shape(record: dict[str, Any], spec: dict[str, Any]) -> None:
    name = record["name"]
    label = f"record {name!r}"
    if record.get("kind") != spec["kind"]:
        raise ValueError(f"{label} has the wrong evidence kind")
    if record.get("result_status") != spec["result_status"]:
        raise ValueError(f"{label} has the wrong retained-result status")
    if set(record.get("files", {})) != spec["files"]:
        raise ValueError(f"{label} does not bind its exact required file set")
    trees = record.get("trees", [])
    actual_trees = {
        tree.get("path"): tree.get("file_count")
        for tree in trees
        if isinstance(tree, dict)
    }
    if actual_trees != spec["trees"] or any(
        tree.get("pattern", "*.json") != "*.json" for tree in trees
    ):
        raise ValueError(f"{label} does not bind its exact required tree set")
    if spec["kind"] == "github_actions_artifact":
        artifact = record.get("artifact", {})
        expected_name = spec["artifact_prefix"] + record.get("source_sha", "")
        if (
            record.get("workflow_file") != ".github/workflows/evidence-v2.yml"
            or artifact.get("name") != expected_name
        ):
            raise ValueError(f"{label} does not identify its exact workflow artifact")
    else:
        if record.get("generator") != spec["generator"]:
            raise ValueError(f"{label} names the wrong derivation generator")
        if set(record.get("inputs", {})) != spec["inputs"]:
            raise ValueError(f"{label} does not bind its exact derivation inputs")


def _gh_json(endpoint: str) -> Any:
    if shutil.which("gh") is None:
        raise RuntimeError("gh is required for --verify-github")
    completed = subprocess.run(
        ["gh", "api", endpoint],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _artifact_relative_path(logical: str) -> Path:
    path = PurePosixPath(logical)
    try:
        relative = path.relative_to(RESULTS_PREFIX)
    except ValueError as exc:
        raise ValueError(f"evidence file is outside the artifact results root: {logical}") from exc
    return Path(*relative.parts)


def verify_github_records(data: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Recheck immutable Actions metadata and downloaded artifact contents."""
    verified: list[str] = []
    for record in data["records"]:
        if record["kind"] != "github_actions_artifact":
            continue
        label = f"record {record['name']!r}"
        run_id = record["run_id"]
        run = _gh_json(f"repos/{REPOSITORY}/actions/runs/{run_id}")
        expected_conclusion = (
            "failure" if record["conclusion"] == "expected_failure" else "success"
        )
        if (
            run.get("head_sha") != record["source_sha"]
            or run.get("html_url") != record["run_url"]
            or run.get("path") != record["workflow_file"]
            or run.get("conclusion") != expected_conclusion
        ):
            raise ValueError(f"{label} disagrees with live Actions run metadata")
        remote = _gh_json(f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts")
        matches = [
            artifact for artifact in remote.get("artifacts", [])
            if artifact.get("id") == record["artifact"]["id"]
        ]
        if len(matches) != 1:
            raise ValueError(f"{label} artifact ID is missing or ambiguous on GitHub")
        artifact = matches[0]
        expected_metadata = (
            record["artifact"]["name"],
            record["artifact"]["digest"],
            record["artifact"]["size_in_bytes"],
            False,
        )
        actual_metadata = (
            artifact.get("name"), artifact.get("digest"),
            artifact.get("size_in_bytes"), artifact.get("expired"),
        )
        if actual_metadata != expected_metadata:
            raise ValueError(f"{label} disagrees with live artifact metadata")

        with tempfile.TemporaryDirectory(prefix="coldplate-evidence-") as temporary:
            destination = Path(temporary)
            subprocess.run(
                [
                    "gh", "run", "download", str(run_id), "--repo", REPOSITORY,
                    "--name", record["artifact"]["name"], "--dir", str(destination),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            for logical, expected_hash in record["files"].items():
                downloaded = destination / _artifact_relative_path(logical)
                if not downloaded.is_file() or sha256_file(downloaded) != expected_hash.removeprefix("sha256:"):
                    raise ValueError(f"{label} downloaded bytes do not match {logical}")
            for tree in record.get("trees", []):
                downloaded = destination / _artifact_relative_path(tree["path"])
                actual_hash, actual_count = tree_digest(downloaded, tree.get("pattern", "*.json"))
                if (actual_count, actual_hash) != (tree["file_count"], tree["sha256"]):
                    raise ValueError(f"{label} downloaded tree bytes do not match {tree['path']}")
        verified.append(record["name"])
    return {"github_records_verified": sorted(verified)}


def validate_manifest(
    path: Path = DEFAULT_MANIFEST,
    *,
    root: Path = ROOT,
    verify_github: bool = False,
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "coldplate-evidence-provenance-v1":
        raise ValueError("unknown evidence provenance schema")
    if data.get("repository") != REPOSITORY:
        raise ValueError("evidence provenance names the wrong repository")
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("evidence provenance contains no records")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("evidence record must be an object")
        name, kind = record.get("name"), record.get("kind")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("evidence record names must be nonempty and unique")
        names.add(name)
        label = f"record {name!r}"
        if kind == "github_actions_artifact":
            run_id = record.get("run_id")
            source_sha = record.get("source_sha")
            artifact = record.get("artifact")
            workflow_file = record.get("workflow_file")
            if (
                isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or run_id <= 0
                or record.get("run_url")
                != f"https://github.com/{REPOSITORY}/actions/runs/{run_id}"
                or not isinstance(source_sha, str)
                or COMMIT_RE.fullmatch(source_sha) is None
                or not isinstance(workflow_file, str)
                or not _repo_path(root, workflow_file).is_file()
                or record.get("conclusion") not in {"success", "expected_failure"}
                or not isinstance(artifact, dict)
                or isinstance(artifact.get("id"), bool)
                or not isinstance(artifact.get("id"), int)
                or artifact["id"] <= 0
                or not isinstance(artifact.get("name"), str)
                or not artifact["name"]
                or source_sha not in artifact["name"]
                or not isinstance(artifact.get("digest"), str)
                or SHA256_RE.fullmatch(artifact["digest"]) is None
                or isinstance(artifact.get("size_in_bytes"), bool)
                or not isinstance(artifact.get("size_in_bytes"), int)
                or artifact["size_in_bytes"] <= 0
            ):
                raise ValueError(f"{label} has invalid Actions provenance")
        elif kind == "derived":
            generator = record.get("generator")
            if not isinstance(generator, str) or not _repo_path(root, generator).is_file():
                raise ValueError(f"{label} has no committed generator")
            _validate_files(root, record.get("inputs"), f"{label} inputs")
        else:
            raise ValueError(f"{label} has unknown kind {kind!r}")
        _validate_files(root, record.get("files"), f"{label} files")
        _validate_trees(root, record.get("trees", []), label)
    if names != set(EXPECTED_RECORDS):
        missing = sorted(set(EXPECTED_RECORDS) - names)
        extra = sorted(names - set(EXPECTED_RECORDS))
        raise ValueError(f"evidence provenance record coverage mismatch: missing={missing}, extra={extra}")
    by_name = {record["name"]: record for record in records}
    for name, spec in EXPECTED_RECORDS.items():
        _validate_expected_shape(by_name[name], spec)
    report: dict[str, Any] = {"records": len(records), "names": sorted(names)}
    if verify_github:
        report.update(verify_github_records(data, root=root))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify-github", action="store_true",
        help="query and download each Actions artifact with gh before accepting it",
    )
    args = parser.parse_args()
    print(json.dumps(
        validate_manifest(args.manifest, verify_github=args.verify_github),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
