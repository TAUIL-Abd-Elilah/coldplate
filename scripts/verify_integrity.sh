#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Independently confirm the two claims the builds enforce, from outside the
# build. A check that only runs inside its own build is worth less than one
# anybody can re-run against the published image.
#
#   1. The C++ and Fortran components contain no autodiff framework at all.
#   2. The Fortran library really was differentiated by Enzyme: its linked
#      symbols include cosh, which appears in no source file. It is the
#      generated derivative of the tanh in the Peclet weighting.
#
#   usage:  scripts/verify_integrity.sh
set -uo pipefail

PROBE='import importlib.util as u
mods = ("jax", "torch", "tensorflow", "autograd", "casadi")
bad = [m for m in mods if u.find_spec(m)]
print("  importable AD frameworks:", bad if bad else "none")
raise SystemExit(1 if bad else 0)'

fail=0

for img in stokes_brinkman thermal_fortran; do
    echo "=== $img: no autodiff framework ==="
    if docker run --rm --entrypoint /python-env/bin/python3 "$img" -c "$PROBE"; then
        echo "  PASS"
    else
        echo "  FAIL"
        fail=1
    fi
done

echo "=== thermal_fortran: Enzyme actually differentiated the Fortran ==="
# nm prints versioned symbols such as "cosh@GLIBC_2.2.5", so strip the suffix
# before matching -- an exact-match pattern silently misses every libm symbol.
syms=$(docker run --rm --entrypoint bash thermal_fortran -c \
    'nm -D --undefined-only /tesseract/lib/libthermal_ad.so | awk "{print \$2}" | sed "s/@.*//"' 2>/dev/null)
echo "$syms" | grep -q '^tanh$' && echo "  tanh   present (called by the residual)"
if echo "$syms" | grep -q '^cosh$'; then
    echo "  cosh   present (d/dx tanh = 1/cosh^2 -- generated, in no source file)"
    echo "  PASS"
else
    echo "  cosh   MISSING -- the Enzyme pass did not run"
    echo "  FAIL"
    fail=1
fi

echo
[ "$fail" -eq 0 ] && echo "all integrity claims verified" || echo "INTEGRITY CHECK FAILED"
exit "$fail"
