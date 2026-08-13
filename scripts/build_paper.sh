#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Render PAPER.md to a typeset PDF.
#
# Markdown -> HTML -> PDF via WeasyPrint rather than through LaTeX: the paper is
# carried by wide result tables that CSS controls far better than LaTeX's
# defaults, and the source keeps Unicode (10⁻¹², γ, Φ, ×) so it stays readable
# as plain Markdown on GitHub, which pdfLaTeX rejects outright.
#
# pandoc emits a *fragment* and we supply the surrounding document ourselves.
# Using --standalone instead injects its own title block above the paper's real
# title and wraps sections in divs, which silently breaks sibling selectors in
# the stylesheet.
#
#   apt-get install pandoc weasyprint fonts-dejavu
#   usage:  scripts/build_paper.sh [output.pdf]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/PAPER.pdf}"
FRAG="$(mktemp -t coldplate-frag-XXXXXX.html)"
HTML="$(mktemp -t coldplate-paper-XXXXXX.html)"
trap 'rm -f "$FRAG" "$HTML"' EXIT

pandoc "$ROOT/PAPER.md" --from=markdown+pipe_tables --to=html5 -o "$FRAG"

{
    printf '<!doctype html><html><head><meta charset="utf-8">'
    printf '<title>Coldplate</title></head><body>\n'
    cat "$FRAG"
    printf '\n</body></html>\n'
} > "$HTML"

weasyprint "$HTML" "$OUT" --stylesheet "$ROOT/scripts/paper.css"

echo "wrote $OUT"
