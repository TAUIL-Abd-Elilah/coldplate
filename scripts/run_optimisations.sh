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
PY="${PYTHON:-python}"

cd "$ROOT/orchestrator"
for mode in composed one_way; do
    echo "=============== $mode (N=$N, iters=$ITERS) ==============="
    extra=""
    [ "$mode" = "composed" ] && extra="--diagnose 6"
    "$PY" -u optimize.py --N "$N" --iters "$ITERS" --mode "$mode" $extra
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
done
echo "optimisation runs complete"
