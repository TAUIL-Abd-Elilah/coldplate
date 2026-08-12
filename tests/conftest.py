# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures.

These tests deliberately exercise the Tesseract API modules *directly* rather
than through served containers. The endpoints are ordinary Python functions, so
this covers all the numerics and all the derivative code without needing Docker
-- which is what lets the whole suite run in CI.

Container-level composition is covered separately by
`orchestrator/compare_to_reference.py`, which does need Docker.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EIGEN_CANDIDATES = [
    Path("/usr/include/eigen3"),
    Path("/usr/local/include/eigen3"),
    ROOT.parent / "ref" / "eigen-3.4.0",
]


def _add_path(p: Path) -> None:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_tesseract_api(name: str):
    """Import a Tesseract's api module under a distinct name.

    All three are called `tesseract_api.py`, so they cannot simply be imported.
    Registering in sys.modules before executing matters: these modules use
    `from __future__ import annotations`, so pydantic resolves `Differentiable`
    lazily against the module namespace and fails with "not fully defined" if
    the module is not findable there.
    """
    import importlib.util

    mod_name = f"{name}_api"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    path = ROOT / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session", autouse=True)
def _paths():
    _add_path(ROOT / "prototype")
    _add_path(ROOT / "tesseracts" / "stokes_brinkman")
    _add_path(ROOT / "tesseracts" / "thermal_advdiff")
    _add_path(ROOT / "tesseracts" / "material_map")


@pytest.fixture(scope="session")
def stokes_lib(_paths):
    """Compile the C++ solver if the shared library is not already built."""
    libdir = ROOT / "tesseracts" / "stokes_brinkman" / "lib"
    for name in ("libstokes_brinkman.so", "stokes_brinkman.dll"):
        if (libdir / name).exists():
            return libdir / name

    eigen = next((p for p in EIGEN_CANDIDATES if (p / "Eigen" / "Sparse").exists()), None)
    if eigen is None:
        pytest.skip("Eigen headers not found; cannot build the C++ solver")
    if shutil.which("g++") is None:
        pytest.skip("g++ not available; cannot build the C++ solver")

    libdir.mkdir(parents=True, exist_ok=True)
    out = libdir / "libstokes_brinkman.so"
    subprocess.run(
        ["g++", "-O2", "-DNDEBUG", "-fPIC", "-shared", "-fvisibility=hidden",
         f"-I{eigen}",
         str(ROOT / "tesseracts" / "stokes_brinkman" / "src" / "stokes_brinkman.cpp"),
         "-o", str(out)],
        check=True,
    )
    return out
