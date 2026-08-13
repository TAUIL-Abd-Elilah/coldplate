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
PY="${PYTHON:-python}"

cd "$ROOT/orchestrator"
for backend in thermal_advdiff thermal_fortran; do
    echo "################ $backend ################"
    "$PY" -u validate_pipeline.py "$N" "$backend"
    # reap served containers between runs so they do not accumulate
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
done
echo "both backends validated"
