#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the headline gradient validation through BOTH thermal backends, so
# every number in the README is shown to be independent of which derivative
# technology produced it: JAX autodiff in one case, an Enzyme compiler pass
# over Fortran in the other.
#
# Expects all four Tesseracts to be built already (see the README).
#
#   usage:  scripts/validate_both_backends.sh [N]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-16}"
# Resolve the interpreter. Defaulting to "python" is wrong on most Linux
# systems, where only python3 exists -- and with a non-fatal set this script
# then reported success while running nothing at all.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3
    elif command -v python >/dev/null 2>&1; then PY=python
    else echo "no python interpreter found; set PYTHON=..." >&2; exit 1; fi
fi

cd "$ROOT/orchestrator"

# Tesseract normally tears down what it serves. If a Python process crashes,
# clean up only containers descended from this project's four images. Never
# touch unrelated containers on the reviewer's machine.
cleanup_project_containers() {
    for image in material_map stokes_brinkman thermal_advdiff thermal_fortran; do
        ids="$(docker ps -q --filter "ancestor=$image" 2>/dev/null || true)"
        [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true
    done
}
trap cleanup_project_containers EXIT

rc=0
for backend in thermal_advdiff thermal_fortran; do
    echo "################ $backend ################"
    if ! "$PY" -u validate_pipeline.py "$N" "$backend"; then
        echo "  $backend FAILED"
        rc=1
    fi
    cleanup_project_containers
done

# Report the real outcome. An earlier version printed success unconditionally,
# so a missing interpreter looked exactly like a passing validation.
if [ "$rc" -eq 0 ]; then
    echo "both backends validated"
else
    echo "VALIDATION FAILED"
fi
exit "$rc"
