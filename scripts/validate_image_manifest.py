#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the exact four OCI digest references consumed by judge tooling."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

COMPONENTS = (
    "stokes_brinkman",
    "thermal_advdiff",
    "thermal_fortran",
    "material_map",
)


def parse_manifest(path: str | Path) -> dict[str, str]:
    """Return component-to-reference mapping or reject the whole manifest."""
    source = Path(path)
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.count("=") != 1:
            raise ValueError(f"{source}:{line_number}: expected component=reference")
        component, reference = line.split("=", 1)
        if component not in COMPONENTS:
            raise ValueError(f"{source}:{line_number}: unknown component {component!r}")
        if component in rows:
            raise ValueError(f"{source}:{line_number}: duplicate component {component!r}")
        expected = re.compile(
            rf"^ghcr\.io/[a-z0-9](?:[a-z0-9-]{{0,38}})/"
            rf"coldplate-{re.escape(component)}@sha256:[0-9a-f]{{64}}$"
        )
        if not expected.fullmatch(reference):
            raise ValueError(f"{source}:{line_number}: invalid OCI digest reference")
        rows[component] = reference

    missing = [component for component in COMPONENTS if component not in rows]
    if missing:
        raise ValueError(f"{source}: missing components: {', '.join(missing)}")
    return {component: rows[component] for component in COMPONENTS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = parse_manifest(args.manifest)
    for component, reference in rows.items():
        print(f"{component}={reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
