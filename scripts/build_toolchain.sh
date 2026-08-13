#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Build the compiler toolchain image the Fortran/Enzyme Tesseract is built on:
# flang + LLVM 19 + the Enzyme plugin.
#
# Split out of the Tesseract build because it pulls ~200 MB of compiler across
# the network. Building it separately means that cost is paid once and cached.
# Retries because large downloads on some networks arrive corrupted, which
# surfaces as an apt hash mismatch or a TLS "bad record mac" rather than a
# clean failure.
#
#   usage:  scripts/build_toolchain.sh [attempts]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATTEMPTS="${1:-4}"
TAG=coldplate-enzyme-toolchain:1.0

for i in $(seq 1 "$ATTEMPTS"); do
    echo "=== toolchain build attempt $i/$ATTEMPTS ==="
    docker build -t "$TAG" "$ROOT/tesseracts/thermal_fortran/toolchain" 2>&1 | tail -14
    if docker image inspect "$TAG" >/dev/null 2>&1; then
        echo "built $TAG"
        docker run --rm "$TAG" bash -lc \
            'opt --version | grep -o "LLVM version [0-9.]*"; flang-new --version | head -1'
        exit 0
    fi
    sleep 20
done

echo "toolchain build failed after $ATTEMPTS attempts"
exit 1
