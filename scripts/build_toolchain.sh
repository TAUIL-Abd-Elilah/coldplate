#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Fortran+Enzyme toolchain base image, retrying on the corrupted
# large downloads this network intermittently produces. Once it succeeds the
# image is cached and Tesseract rebuilds are fast and offline.
set -uo pipefail

SRC=/mnt/d/Competition/pasteurlabs/coldplate/tesseracts/thermal_fortran/toolchain
WORK=/root/toolchain
TAG=coldplate-enzyme-toolchain:1.0

systemctl start docker >/dev/null 2>&1 || true
mkdir -p "$WORK"
tr -d '\r' < "$SRC/Dockerfile" > "$WORK/Dockerfile"

for i in 1 2 3 4 5; do
    echo "=== toolchain build attempt $i ==="
    docker build -t "$TAG" "$WORK" 2>&1 | tail -14
    if docker image inspect "$TAG" >/dev/null 2>&1; then
        echo "TOOLCHAIN_OK"
        docker run --rm "$TAG" bash -lc 'opt --version | head -2; (flang-new --version || flang --version) | head -1'
        exit 0
    fi
    sleep 20
done

echo "TOOLCHAIN_FAILED"
exit 1
