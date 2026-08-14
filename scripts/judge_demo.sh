#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Safe, reviewer-facing golden path. It builds missing project images, verifies
# the derivative implementations, and proves that swapping JAX for
# Fortran/Enzyme leaves the full coupled gradient unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N=8
BUILD=1

usage() {
    echo "usage: scripts/judge_demo.sh [--grid N] [--no-build]"
    echo "  first source build: roughly 10-30 min, 8-10 GB free disk recommended"
    echo "  warm run with images present: roughly 1-3 min"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --grid) N="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ "$(uname -s)" = "Linux" ] || {
    echo "This demo requires Linux containers. On Windows, run it inside WSL2." >&2
    exit 2
}
case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "The published Tesseracts target linux/amd64; found $(uname -m)." >&2; exit 2 ;;
esac

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "the Docker daemon is not reachable" >&2; exit 2; }

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3
    elif command -v python >/dev/null 2>&1; then PY=python
    else echo "Python 3.12+ is required" >&2; exit 2; fi
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
"$PY" -c 'import tesseract_core, tesseract_jax' 2>/dev/null || {
    echo "Install the driver first: $PY -m pip install -r requirements-orchestrator.txt" >&2
    exit 2
}
command -v tesseract >/dev/null || { echo "the tesseract CLI is not installed" >&2; exit 2; }

cleanup_project_containers() {
    for image in material_map stokes_brinkman thermal_advdiff thermal_fortran; do
        ids="$(docker ps -q --filter "ancestor=$image" 2>/dev/null || true)"
        [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true
    done
}
trap cleanup_project_containers EXIT

missing=0
for image in material_map stokes_brinkman thermal_advdiff thermal_fortran; do
    docker image inspect "$image" >/dev/null 2>&1 || missing=1
done

if [ "$missing" -eq 1 ]; then
    [ "$BUILD" -eq 1 ] || {
        echo "project images are missing; rerun without --no-build" >&2
        exit 2
    }
    docker image inspect coldplate-enzyme-toolchain:1.0 >/dev/null 2>&1 \
        || bash "$ROOT/scripts/build_toolchain.sh"
    for component in stokes_brinkman thermal_advdiff thermal_fortran material_map; do
        tesseract build "$ROOT/tesseracts/$component"
    done
fi

echo "=== 1/2 derivative provenance ==="
bash "$ROOT/scripts/verify_integrity.sh"

echo
echo "=== 2/2 drop-in backend and end-to-end gradient ==="
cd "$ROOT/orchestrator"
"$PY" -u compare_thermal_backends.py "$N"

echo
echo "PASS: three active components, one drop-in thermal backend, and numerically matching gradients"
