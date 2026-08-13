#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Report the exact package versions inside each built Tesseract, plus the
# driving environment. Used to pin requirements to what was actually built
# rather than to whatever resolves today.
#
#   usage:  scripts/capture_versions.sh
set -uo pipefail

# Resolve the interpreter. Defaulting to "python" is wrong on most Linux
# systems, where only python3 exists -- and with a non-fatal set this script
# then reported success while running nothing at all.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3
    elif command -v python >/dev/null 2>&1; then PY=python
    else echo "no python interpreter found; set PYTHON=..." >&2; exit 1; fi
fi
PROBE='import importlib.metadata as m
for p in ("jax","jaxlib","numpy","scipy","torch"):
    try: print("  %s==%s" % (p, m.version(p)))
    except Exception: pass'

echo "=== orchestrator (drives the composition) ==="
"$PY" -c 'import importlib.metadata as m
for p in ("tesseract-core","tesseract-jax","jax","jaxlib","numpy","scipy","matplotlib"):
    try: print("  %s==%s" % (p, m.version(p)))
    except Exception: print("  %s: missing" % p)'

for img in stokes_brinkman thermal_advdiff thermal_fortran material_map; do
    echo
    echo "=== $img ==="
    docker run --rm --entrypoint /python-env/bin/python3 "$img" -c "$PROBE" 2>/dev/null \
        || echo "  (image not built)"
done

echo
echo "=== toolchain image ==="
docker run --rm coldplate-enzyme-toolchain:1.0 bash -lc \
    'opt --version | grep -o "LLVM version [0-9.]*" | sed "s/^/  /"; flang-new --version | head -1 | sed "s/^/  /"' \
    2>/dev/null || echo "  (not built)"
