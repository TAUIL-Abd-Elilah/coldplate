#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Render PAPER.md to PDF. Needs pandoc and XeLaTeX:
#   apt-get install pandoc texlive-xetex texlive-latex-recommended fonts-dejavu
#
# XeLaTeX rather than pdfLaTeX because the paper uses Unicode superscripts and
# multiplication signs (10⁻¹², ×) so it stays readable as plain Markdown;
# pdfLaTeX's default encoding rejects those outright.
#
#   usage:  scripts/build_paper.sh [output.pdf]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/PAPER.pdf}"

pandoc "$ROOT/PAPER.md" -o "$OUT" \
    --pdf-engine=xelatex \
    -V geometry:a4paper \
    -V geometry:margin=2.2cm \
    -V fontsize=10pt \
    -V mainfont="DejaVu Serif" \
    -V sansfont="DejaVu Sans" \
    -V monofont="DejaVu Sans Mono" \
    -V colorlinks=true \
    -V linkcolor=RoyalBlue \
    -V urlcolor=RoyalBlue \
    --highlight-style=tango

echo "wrote $OUT"
