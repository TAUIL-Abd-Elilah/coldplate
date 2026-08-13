#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Build a Tesseract, retrying on transient network failures. The Enzyme image
# pulls an LLVM toolchain, LFortran via micromamba and the Enzyme plugin, so it
# touches several remote hosts and is much more exposed to a flaky link than
# our other images.
#   usage: build_retry.sh <dir> <image-name-substring> [attempts]
set -uo pipefail

DIR="${1:?need a tesseract directory}"
NAME="${2:?need an image name substring}"
ATTEMPTS="${3:-4}"

systemctl start docker >/dev/null 2>&1 || true

for i in $(seq 1 "$ATTEMPTS"); do
    echo "--- attempt $i/$ATTEMPTS ---"
    /root/venv/bin/tesseract build "$DIR" 2>&1 | tail -8
    if docker images --format '{{.Repository}}' | grep -q "$NAME"; then
        echo "BUILD_OK: $NAME"
        exit 0
    fi
    sleep 15
done

echo "BUILD_FAILED after $ATTEMPTS attempts"
exit 1
