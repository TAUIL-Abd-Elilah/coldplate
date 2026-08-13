#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the headline gradient validation through BOTH thermal backends, so
# every number in the README is shown to be independent of which derivative
# technology produced it.
set -uo pipefail

N="${1:-16}"
WORK=/root/coldplate

systemctl start docker >/dev/null 2>&1 || true
rsync -a --exclude '__pycache__' --exclude '*.dll' --exclude '*.obj' \
    --exclude '*.lib' --exclude '*.exp' --exclude 'results*' \
    /mnt/d/Competition/pasteurlabs/coldplate/ "$WORK/"

cd "$WORK/orchestrator"
for backend in thermal_advdiff thermal_fortran; do
    echo "################ $backend ################"
    /root/venv/bin/python -u validate_pipeline.py "$N" "$backend" \
        2>&1 | grep --line-buffered -vE 'NVIDIA GPU|^ *newton'
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
done
echo BOTH_BACKENDS_DONE
