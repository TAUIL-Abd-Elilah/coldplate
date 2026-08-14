#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Pull the exact OCI manifests recorded in the submission release and retag
# them to the local names consumed by the orchestrator. No registry login is
# performed: success therefore proves the packages are anonymously accessible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${COLDPLATE_GITHUB_OWNER:-TAUIL-Abd-Elilah}"
REPOSITORY="${COLDPLATE_GITHUB_REPOSITORY:-coldplate}"
RELEASE_TAG="${COLDPLATE_RELEASE_TAG:-submission-2026}"
MANIFEST="${1:-https://github.com/$OWNER/$REPOSITORY/releases/download/$RELEASE_TAG/image-digests.txt}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

temporary="$(mktemp -d)"
cleanup() {
    rm -f -- "$temporary/image-digests.txt"
    rmdir -- "$temporary" 2>/dev/null || true
}
trap cleanup EXIT

if [[ "$MANIFEST" = http://* || "$MANIFEST" = https://* ]]; then
    command -v curl >/dev/null || { echo "curl is required for a release URL" >&2; exit 2; }
    curl --fail --location --silent --show-error "$MANIFEST" -o "$temporary/image-digests.txt"
    manifest_path="$temporary/image-digests.txt"
else
    manifest_path="$MANIFEST"
fi
[ -f "$manifest_path" ] || { echo "digest manifest not found: $manifest_path" >&2; exit 2; }

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3
    elif command -v python >/dev/null 2>&1; then PY=python
    else echo "Python 3 is required to validate the digest manifest" >&2; exit 2; fi
fi
"$PY" "$ROOT/scripts/validate_image_manifest.py" "$manifest_path" >/dev/null

expected="stokes_brinkman thermal_advdiff thermal_fortran material_map"
seen=""
while IFS='=' read -r component reference; do
    [ -n "$component" ] || continue
    case " $expected " in
        *" $component "*) ;;
        *) echo "unexpected component in digest manifest: $component" >&2; exit 2 ;;
    esac
    case "$reference" in
        ghcr.io/*/coldplate-"$component"@sha256:????????????????????????????????????????????????????????????????) ;;
        *) echo "invalid digest reference for $component: $reference" >&2; exit 2 ;;
    esac
    case " $seen " in
        *" $component "*) echo "duplicate component in digest manifest: $component" >&2; exit 2 ;;
    esac
    docker pull "$reference"
    docker tag "$reference" "$component:latest"
    seen="$seen $component"
done < "$manifest_path"

for component in $expected; do
    case " $seen " in
        *" $component "*) ;;
        *) echo "digest manifest is missing $component" >&2; exit 2 ;;
    esac
done

echo "PASS: pulled and locally tagged four digest-pinned submission images"
