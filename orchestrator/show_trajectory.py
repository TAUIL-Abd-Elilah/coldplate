# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Report how wrong the naive gradient is along the optimisation trajectory.

The optimisation succeeds with the naive gradient at this operating point,
which on its own looks like an argument against needing the composed one. This
is the resolution: the coupling weakens as the design solidifies, because solid
material blocks the flow. So the naive gradient is badly wrong exactly where
the design is still uniform, and becomes accurate once it is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path="results/history_diag_N48.json"):
    hist = json.loads(Path(path).read_text())
    rows = [r for r in hist if "naive_rel_err" in r]
    if not rows:
        print("no diagnostics in that history file")
        return 1

    hdr = f"{'iter':>5} {'J':>8} {'loop gain':>10} {'naive err':>10} {'cos':>8} {'wrong sign':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['iter']:5d} {r['J']:8.4f} {r['loop_gain']:10.4f} "
            f"{r['naive_rel_err']:10.4f} {r['naive_cos']:8.4f} "
            f"{100*r['naive_sign_flip']:10.0f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/history_diag_N48.json"))
