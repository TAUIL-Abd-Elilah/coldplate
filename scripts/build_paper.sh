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
# FontTools otherwise writes the current time into subset-font ``head`` tables,
# making byte-identical paper builds differ by one compressed byte or more.
# A fixed hackathon timestamp plus deterministic hash iteration makes the
# tracked PDF reproducible across clean CI runs.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1785715200}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
FRAG="$(mktemp -t coldplate-frag-XXXXXX.html)"
HTML="$(mktemp -t coldplate-paper-XXXXXX.html)"
trap 'rm -f "$FRAG" "$HTML"' EXIT

pandoc "$ROOT/PAPER.md" --from=markdown+pipe_tables --to=html5 -o "$FRAG"

{
    printf '<!doctype html><html><head><meta charset="utf-8">'
    printf '<meta name="author" content="TAUIL-Abd-Elilah">'
    printf '<meta name="description" content="Tesseract Hackathon 2026 multi-physics submission">'
    printf '<title>When can you differentiate coupled simulation components separately?</title></head><body>\n'
    cat "$FRAG"
    printf '\n</body></html>\n'
} > "$HTML"

# --base-url is not optional: the HTML lives in a temp directory, so without it
# WeasyPrint resolves the figures relative to /tmp and silently emits a PDF with
# every image missing rather than failing.
weasyprint "$HTML" "$OUT" --stylesheet "$ROOT/scripts/paper.css" --base-url "$ROOT/"

echo "wrote $OUT"
