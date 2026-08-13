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

PY="${PYTHON:-python}"
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
