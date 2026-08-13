#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Full optimisation runs: once driven by the exact composed gradient, once by
# the strong naive one (full chain, feedback loop cut). Same seed, same
# schedule, so the only difference is the gradient.
set -uo pipefail

N="${1:-48}"
ITERS="${2:-120}"
WORK=/root/coldplate

systemctl start docker >/dev/null 2>&1 || true
rsync -a --exclude '__pycache__' --exclude '*.dll' --exclude '*.obj' \
    --exclude '*.lib' --exclude '*.exp' --exclude 'results*' \
    /mnt/d/Competition/pasteurlabs/coldplate/ "$WORK/"

cd "$WORK/orchestrator"
for mode in composed one_way; do
    echo "=============== $mode (N=$N, iters=$ITERS) ==============="
    # The composed run also records how wrong the naive gradient is at each
    # design it passes through, so one run feeds fig1, fig4 and fig6.
    extra=""
    [ "$mode" = "composed" ] && extra="--diagnose 6"
    /root/venv/bin/python -u optimize.py --N "$N" --iters "$ITERS" --mode "$mode" $extra \
        2>&1 | grep --line-buffered -vE 'NVIDIA GPU|^ *newton'
    # reap containers between runs so they do not accumulate
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
done
echo ALL_RUNS_DONE
