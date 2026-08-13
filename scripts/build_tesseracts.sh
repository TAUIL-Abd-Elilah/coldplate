#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Sync sources from the Windows drive into ext4 and build all three Tesseracts.
set -euo pipefail

WORK=/root/coldplate
SRC=/mnt/d/Competition/pasteurlabs/coldplate

systemctl start docker >/dev/null 2>&1 || true

rsync -a --delete \
    --exclude '__pycache__' --exclude '*.dll' --exclude '*.obj' \
    --exclude '*.lib' --exclude '*.exp' \
    "$SRC/" "$WORK/"

cd "$WORK"
for t in "$@"; do
    echo "=== building $t ==="
    /root/venv/bin/tesseract build "tesseracts/$t" 2>&1 | tail -6
done

echo "=== images ==="
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E 'stokes|thermal|material' || true
