#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Full optimisation runs: once driven by the exact composed gradient, once by
# the strong naive one (full chain, feedback loop cut). Same seed, same
# schedule, so the only difference is the gradient.
#
# The composed run also records how wrong the naive gradient is at each design
# it passes through, so one run feeds figures 1, 4 and 6.
#
#   usage:  scripts/run_optimisations.sh [N] [iters]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-96}"
ITERS="${2:-120}"
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
for mode in composed one_way; do
    echo "=============== $mode (N=$N, iters=$ITERS) ==============="
    extra=""
    [ "$mode" = "composed" ] && extra="--diagnose 6"
    if ! "$PY" -u optimize.py --N "$N" --iters "$ITERS" --mode "$mode" $extra; then
        echo "  $mode FAILED"
        docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
        exit 1
    fi
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
done
echo "optimisation runs complete"
