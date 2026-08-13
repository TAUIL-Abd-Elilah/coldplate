#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Run the Fortran -> Enzyme compile pipeline directly in the toolchain image
# with the sources bind-mounted. Much faster to iterate on than going through
# a full `tesseract build`, and it shows the real compiler diagnostics.
set -uo pipefail

SRC=/mnt/d/Competition/pasteurlabs/coldplate/tesseracts/thermal_fortran/src

docker run --rm -v "$SRC":/src:ro coldplate-enzyme-toolchain:1.0 bash -c '
set -x
mkdir -p /out
flang-new -S -emit-llvm /src/thermal_residual.f90 -o /tmp/thermal.ll || { echo "STAGE=flang"; exit 1; }
opt -O1 -S /tmp/thermal.ll -o /tmp/thermal_opt.ll || { echo "STAGE=opt1"; exit 1; }
clang -emit-llvm -S -O1 /src/wrapper.c -o /tmp/wrapper.ll || { echo "STAGE=clang"; exit 1; }
llvm-link /tmp/wrapper.ll /tmp/thermal_opt.ll -S -o /tmp/combined.ll || { echo "STAGE=link"; exit 1; }
opt --load-pass-plugin="${ENZYME_LIB}" -passes=enzyme -S /tmp/combined.ll -o /tmp/ad.ll || { echo "STAGE=enzyme"; exit 1; }
flang-new -shared -fPIC -O2 /tmp/ad.ll -o /out/libthermal_ad.so -lm || { echo "STAGE=shared"; exit 1; }
set +x
echo "=== exported symbols ==="
nm -D --defined-only /out/libthermal_ad.so | grep -E "th_forward|th_jvp|th_vjp" || echo "SYMBOLS MISSING"
echo "=== undefined symbols ==="
nm -D --undefined-only /out/libthermal_ad.so | head -20
echo COMPILE_OK
'
