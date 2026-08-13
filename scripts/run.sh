#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Sync sources into ext4 and run a script from the orchestrator directory.
#   usage: run.sh <script.py> [args...]
set -euo pipefail

WORK=/root/coldplate
SRC=/mnt/d/Competition/pasteurlabs/coldplate

systemctl start docker >/dev/null 2>&1 || true

# Deliberately no pkill of previous runs here: this script is invoked as
# `run.sh <script.py>`, so any pattern matching the target script also matches
# this shell's own command line and pkill would terminate its own parent.

rsync -a \
    --exclude '__pycache__' --exclude '*.dll' --exclude '*.obj' \
    --exclude '*.lib' --exclude '*.exp' --exclude 'results' \
    "$SRC/" "$WORK/"

cd "$WORK/orchestrator"
exec /root/venv/bin/python -u "$@"
