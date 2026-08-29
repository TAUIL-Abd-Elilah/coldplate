# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run Tesseract's own gradient checker against all four components.

Everything else in this repository checks the *composition*: the coupled
adjoint against finite differences, the two thermal backends against each
other, the whole pipeline against a monolithic reference. Those are our checks,
written by us, and a reviewer is entitled to ask whether the components would
survive somebody else's.

`tesseract_core.runtime.testing.finite_differences.check_gradients` -- the
function the `tesseract-runtime check-gradients` CLI calls -- is somebody
else's. It ships
inside Tesseract, samples random (input index, output index) pairs, and
compares each declared derivative endpoint against a central difference taken
through `apply`. It knows nothing about cold plates and we did not write it.

Two things make this more than a formality here:

  * The four components obtain their derivatives four different ways -- a
    hand-derived discrete adjoint in C++, JAX autodiff, Enzyme compiler AD over
    Fortran, and torch.autograd. The checker is indifferent to all of that.
  * The state it checks at is the converged coupled fixed point, obtained by
    actually running the loop. That is where the composed adjoint evaluates its
    component derivatives, so it is the state at which being wrong would matter.

## What is recorded, and why it is not just "it passed"

The checker's verdict is `np.allclose(fd_row, endpoint_row, atol=1e-8, rtol=r)`
for a tolerance `r` the caller picks. Neither row depends on `r`, so running at
`rtol=0` returns both rows for every sampled comparison and the *exact* smallest
`r` that would have passed follows by arithmetic:

    required_r = max over elements of  (|fd - endpoint| - atol) / |endpoint|

That number is what this script records, per input path, rather than a pass at
some tolerance chosen after the fact. The CLI's own default is `rtol=0.1`, which
at float64 measures the sampler rather than the derivative.

It is recorded across a ladder of finite-difference step sizes, because a single
step size is not evidence. A correct derivative shows a *plateau*: too large a
step and the O(eps^2) truncation error dominates, too small and the solver's own
convergence noise divided by eps does. The whole curve is kept so a reader can
see there was a bottom to sit at.

One trap has to be avoided to report any of this. `required_r` folds in the
checker's hard-coded `atol=1e-8`, so a step that shrinks a Jacobian row under
that floor scores a perfect zero for having nothing left to compare. Each rung
therefore also records `max_abs_gradient` and an atol-free
`relative_disagreement`, rungs whose whole row sits under the floor are marked
`vacuous`, and the best rung is chosen on the atol-free number among the rungs
that are not.

## Why the step is scaled per input path

`check-gradients` documents `--eps` as a step "as a fraction of the maximum
absolute value of each input" and then applies it as an absolute number. For
inputs of order one the distinction never surfaces. The Brinkman drag `alpha` is
of order 1e4 at the converged state, so the documented reading of `--eps=1e-6`
asks for a step of ~0.02 and gets 1e-6 -- a relative step of 5e-11, far below
the noise floor of the C++ solver's own convergence, and the checker then
reports 400 failures out of 400 checks against a derivative that agrees with an
independent JAX reimplementation. So this script does per input path what the
docstring says the tool already does. Reported upstream; see `upstream/`.

Usage:
    python orchestrator/check_gradients.py            # audited grid, records
    python orchestrator/check_gradients.py --N 12     # smoke, records nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import jax
import numpy as np
# Imported from the submodule rather than the package: as of tesseract-core
# 1.11.0 `tesseract_core.runtime.testing.__init__` exports nothing, so this is
# the only import path that works. It is the same function the
# `tesseract-runtime check-gradients` CLI calls.
from tesseract_core.runtime.testing.finite_differences import (
    check_gradients as tesseract_check,
)
from tesseract_jax import apply_tesseract

from pipeline import ColdPlate, Params

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]

#: Only a run at this grid rewrites the stored artefact, for the same reason
#: `compare_thermal_backends.py` pins one: a smoke-sized run must never
#: overwrite the measurement the README quotes and `audit_claims.py` checks.
#: The design draw and operating point are `validate_pipeline.py`'s; the grid is
#: smaller than its 20 because tesseract-core 1.11.0 re-runs the whole VJP sweep
#: per sampled index, which makes this quadratic. What is being checked -- whether
#: four derivative implementations agree with a central difference -- does not
#: depend on the grid.
AUDITED_N = 12

#: Relative finite-difference steps, each applied against its own input's
#: magnitude. Spans four decades around the (machine epsilon)^(1/3) ~ 6e-6 that
#: is optimal for a smooth float64 function, because these outputs come from
#: iterative solves whose convergence raises the noise floor and moves the
#: optimum up.
REL_EPS_LADDER = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

#: The checker's own hard-coded absolute floor. Not a caller-settable option,
#: which is why it appears here as a constant rather than a parameter.
ATOL = 1e-8

#: Fixed so the sampled indices are the same on every run. The draw is then
#: reproducible; it is not thereby representative.
SEED = 7

#: Sampled input indices per input path, per endpoint. The checker draws with
#: replacement and compares a whole Jacobian row per draw, so the recorded
#: comparison count is this times the number of differentiable outputs times the
#: number of endpoints. Kept modest because tesseract-core 1.11.0 re-runs the
#: whole VJP sweep for every sampled index (their issue #687, fixed after this
#: pinned release), which makes the check quadratic in this number.
MAX_EVALS = 6

ENDPOINTS = ("jacobian_vector_product", "vector_jacobian_product")

HOW = {
    "stokes_brinkman": "C++/Eigen, hand-derived discrete adjoint",
    "thermal_advdiff": "JAX autodiff",
    "thermal_fortran": "Fortran, Enzyme compiler AD at the LLVM IR level",
    "material_map": "PyTorch, torch.autograd",
}


#: Compiled objects the api modules dlopen at import, and the sources they are
#: built from. The C++ one this script compiles itself. The Fortran one it
#: cannot: that object only exists once Enzyme has run over flang's LLVM IR
#: inside the toolchain image, so it has to come out of the built image.
#:
#: Which makes the source check below load-bearing rather than fussy. A cached
#: image that predates the working tree will hand over a stale object, and a
#: gradient check against the wrong binary is worse than no gradient check --
#: it is a green tick for code nobody is running. Both of this project's images
#: were exactly that stale the first time this ran: the C++ one was missing a
#: symbol the current api calls, and the Fortran one predated the de Vahl Davis
#: boundary conditions.
NATIVE = {
    "stokes_brinkman": {
        "library": "/tesseract/lib/libstokes_brinkman.so",
        "sources": ("src/stokes_brinkman.cpp",),
    },
    "thermal_fortran": {
        "library": "/tesseract/lib/libthermal_ad.so",
        "sources": ("src/thermal_residual.f90", "src/wrapper.c"),
    },
}


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _build_cpp(local: Path) -> bool:
    """Compile the C++ solver from the working tree, as tests/conftest.py does."""
    source = ROOT / "tesseracts" / "stokes_brinkman" / "src" / "stokes_brinkman.cpp"
    eigen = next(
        (path for path in (Path("/usr/include/eigen3"), Path("/usr/local/include/eigen3"))
         if (path / "Eigen" / "Sparse").exists()),
        None,
    )
    if eigen is None or shutil.which("g++") is None:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["g++", "-O2", "-DNDEBUG", "-shared", "-fPIC", "-fvisibility=hidden",
         f"-I{eigen}", str(source), "-o", str(local)],
        check=True,
    )
    print(f"  compiled {local.relative_to(ROOT)} from source", flush=True)
    return True


def _ensure_native_library(name: str) -> None:
    """Put a current compiled object where the api module will find it."""
    spec = NATIVE.get(name)
    if spec is None:
        return
    local = ROOT / "tesseracts" / name / "lib" / Path(spec["library"]).name
    if local.exists():
        return
    if name == "stokes_brinkman" and _build_cpp(local):
        return
    if shutil.which("docker") is None:
        raise SystemExit(
            f"{name} needs {local}, which is built inside its image; docker is "
            "required to extract it"
        )

    container = f"coldplate-extract-{name}"
    _docker("rm", "-f", container)
    created = _docker("create", "--name", container, f"{name}:latest")
    if created.returncode != 0:
        raise SystemExit(
            f"could not create a container from {name}:latest to extract "
            f"{spec['library']}. Build the images first: "
            f"tesseract build tesseracts/{name}\n{created.stderr.strip()}"
        )
    try:
        # Refuse a stale image before trusting anything it contains.
        for relative in spec["sources"]:
            staged = Path(tempfile.gettempdir()) / f"coldplate-{name}-{Path(relative).name}"
            copied = _docker("cp", f"{container}:/tesseract/{relative}", str(staged))
            if copied.returncode != 0:
                raise SystemExit(
                    f"{name}:latest does not carry {relative}; cannot confirm the "
                    f"compiled object matches this working tree"
                )
            here = (ROOT / "tesseracts" / name / relative).read_bytes()
            there = staged.read_bytes()
            staged.unlink(missing_ok=True)
            if here != there:
                raise SystemExit(
                    f"{name}:latest was built from a different {relative} than the "
                    f"one in this working tree, so its compiled object is stale. "
                    f"Rebuild it before checking gradients:\n"
                    f"    tesseract build tesseracts/{name}"
                )
        local.parent.mkdir(parents=True, exist_ok=True)
        copied = _docker("cp", f"{container}:{spec['library']}", str(local))
        if copied.returncode != 0:
            raise SystemExit(f"could not copy {spec['library']} out of "
                             f"{name}:latest:\n{copied.stderr.strip()}")
    finally:
        _docker("rm", "-f", container)
    print(
        f"  extracted {local.relative_to(ROOT)} from {name}:latest "
        f"(sources confirmed identical to the working tree)",
        flush=True,
    )


def _native_digest(name: str) -> dict:
    """Hash the compiled object a native component will be checked against."""
    spec = NATIVE.get(name)
    if spec is None:
        return {}
    local = ROOT / "tesseracts" / name / "lib" / Path(spec["library"]).name
    if not local.exists():
        return {}
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    return {name: {"file": local.name, "bytes": local.stat().st_size,
                   "sha256": digest}}


def _load_api(name: str):
    """Import a Tesseract's api module under a distinct name.

    All four are called `tesseract_api.py`, so they cannot simply be imported.
    Registering in `sys.modules` before executing matters: these modules use
    `from __future__ import annotations`, so pydantic resolves `Differentiable`
    lazily against the module namespace. Same approach as `tests/conftest.py`.
    """
    import importlib.util

    mod_name = f"{name}_api"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = ROOT / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _state(N: int) -> dict[str, dict]:
    """Solve the coupled problem, then hand back each component's own inputs.

    These payloads are what the composed adjoint really feeds these four
    components at the state it differentiates: `alpha` and `k` from the material
    map, `T` at the fixed point, and `(u, v)` from the flow solved at that `T`.
    """
    # Deliberately the same design draw and operating point as
    # `validate_pipeline.py`. A near-uniform draw is not usable: the steady
    # solver does not converge from one at Ra = 3e4, which the README documents
    # as a measured constraint rather than a preference.
    p = Params(Nx=N, Ny=N, Ra=3.0e4)
    rng = np.random.default_rng(0)
    rho_raw = rng.uniform(0.25, 0.75, size=(N, N))

    with ColdPlate(params=p) as cp:
        mat = cp.material(rho_raw)
        alpha, k = np.asarray(mat["alpha"]), np.asarray(mat["k"])
        T_star, info = cp.solve_coupled(mat["alpha"], mat["k"])
        if not info["ok"]:
            raise SystemExit(
                f"the coupled solve did not converge (residual {info['residual']:.3e}); "
                "refusing to check gradients at an unconverged state"
            )
        flow = apply_tesseract(
            cp._t["fluid"],
            {"alpha": mat["alpha"], "T": T_star, "Ra": p.Ra, "Pr": p.Pr,
             "inertia": p.inertia},
        )
        T_star = np.asarray(T_star)
        u, v = np.asarray(flow["u"]), np.asarray(flow["v"])

    thermal = {
        "u": u.tolist(), "v": v.tolist(), "k": k.tolist(),
        "q_chip": p.q_chip, "chip_frac": p.chip_frac,
        "bc_mode": p.bc_mode, "t_hot": p.t_hot,
    }
    return {
        "material_map": {
            "rho_raw": rho_raw.tolist(), "filter_radius": p.filter_radius,
            "beta": p.beta, "eta": p.eta, "penal": p.penal,
            "k_solid": p.k_solid, "k_fluid": p.k_fluid, "alpha_max": p.alpha_max,
        },
        "stokes_brinkman": {
            "alpha": alpha.tolist(), "T": T_star.tolist(),
            "Ra": p.Ra, "Pr": p.Pr, "inertia": p.inertia,
        },
        "thermal_advdiff": thermal,
        "thermal_fortran": dict(thermal),
    }


def _required_rtol(api, inputs: dict, path: str, eps: float) -> dict:
    """What the checker makes of one input path at one step size.

    Runs at `rtol=0`, which makes the checker hand back both rows for every
    comparison not already inside its hard-coded `atol=1e-8`, and inverts its
    own `allclose` test element by element.

    Three numbers come back, and the distinctions matter.

    `required_rtol` is the checker's own verdict, `atol` included. It is what
    the tool would report, and it is the wrong thing to rank steps by: an
    element whose endpoint derivative is near zero blows the ratio up without
    the derivative being wrong.

    `relative_disagreement` divides the largest absolute gap by the largest
    entry of the endpoint's own row, so it is scale-free and cannot be gamed by
    the floor.

    `phantom_rows` counts comparisons where the finite difference is *exactly*
    zero while the endpoint reports a real derivative -- the signature of a
    derivative taken with respect to an input the forward map never reads.
    Those are separated out rather than averaged in, because they are a
    different kind of finding from a tolerance miss, and this project's thermal
    blocks have some.
    """
    worst_rtol = 0.0
    worst_gap = 0.0
    biggest_gradient = 0.0
    relative = 0.0
    relative_live = 0.0
    compared = 0
    outside_atol = 0
    phantom = 0
    live = 0
    for endpoint, failures, num_evals in tesseract_check(
        api, {"inputs": inputs},
        input_paths=[path], endpoints=list(ENDPOINTS),
        max_evals=MAX_EVALS, eps=eps, rtol=0.0, seed=SEED, show_progress=False,
    ):
        compared += num_evals
        for failure in failures:
            outside_atol += 1
            if failure.exception is not None:
                raise SystemExit(
                    f"{path}: {endpoint} raised during the check: {failure.exception}"
                )
            reference = np.asarray(failure.ref_val, dtype=float)
            gradient = np.asarray(failure.grad_val, dtype=float)
            absolute = np.abs(reference - gradient)
            scale = np.abs(gradient)
            row_scale = float(scale.max())
            biggest_gradient = max(biggest_gradient, row_scale)
            worst_gap = max(worst_gap, float(absolute.max()))

            gap = absolute - ATOL
            need = np.where(gap <= 0.0, 0.0,
                            np.divide(gap, scale, out=np.full_like(gap, np.inf),
                                      where=scale > 0.0))
            worst_rtol = max(worst_rtol, float(need.max()))

            if row_scale > 0.0:
                ratio = float(absolute.max()) / row_scale
                relative = max(relative, ratio)
                if not np.any(reference):
                    # The forward map did not move at all, yet a derivative was
                    # reported. Not a tolerance question.
                    phantom += 1
                else:
                    live += 1
                    relative_live = max(relative_live, ratio)
    return {
        "eps": eps,
        "required_rtol": worst_rtol,
        "relative_disagreement": relative,
        "relative_disagreement_live": relative_live,
        "phantom_rows": phantom,
        "live_rows": live,
        "max_abs_gradient": biggest_gradient,
        "max_abs_difference": worst_gap,
        "comparisons": compared,
        "outside_atol": outside_atol,
        # Nothing was reported outside the checker's absolute floor, so every
        # sampled comparison agreed to within 1e-8. That is the strongest
        # outcome available, not an empty one.
        "all_within_atol": outside_atol == 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Tesseract's gradient checker.")
    ap.add_argument("--N", type=int, default=AUDITED_N, help="grid size")
    args = ap.parse_args(argv)
    N = args.N

    print(f"building the coupled state at {N}x{N} ...", flush=True)
    started = time.time()
    payloads = _state(N)

    record: dict = {
        "schema_version": 1,
        "N": int(N),
        "seed": SEED,
        "atol": ATOL,
        "max_evals_per_path": MAX_EVALS,
        "endpoints": list(ENDPOINTS),
        "rel_eps_ladder": list(REL_EPS_LADDER),
        "state": "converged coupled fixed point",
        "checker": "tesseract_core.runtime.testing.finite_differences.check_gradients",
        "step_policy": (
            "eps supplied per input path as rel_eps * max|value| of that input, "
            "because check-gradients applies eps absolutely while documenting it "
            "as a fraction of the input's maximum absolute value"
        ),
        "verdict_policy": (
            "run at rtol=0 and invert the checker's own allclose test, so the "
            "recorded number is the tightest rtol that would have passed rather "
            "than a tolerance picked after seeing the result"
        ),
        "components": {},
        # Which compiled object each native component was checked against, so
        # the record cannot quietly be of a binary nobody is running.
        "native_libraries": {},
    }

    for image, inputs in payloads.items():
        _ensure_native_library(image)
        api = _load_api(image)
        record["native_libraries"].update(_native_digest(image))
        component: dict = {"how_differentiated": HOW[image], "input_paths": {}}

        for name, value in inputs.items():
            if not isinstance(value, list):
                continue
            magnitude = float(np.abs(np.asarray(value, dtype=float)).max())
            ladder = {}
            for rel in REL_EPS_LADDER:
                rung = _required_rtol(api, inputs, name, rel * magnitude)
                ladder[repr(rel)] = rung
                if rung["all_within_atol"]:
                    note = "  all within atol"
                elif rung["phantom_rows"]:
                    note = f"  PHANTOM x{rung['phantom_rows']}"
                else:
                    note = ""
                print(
                    f"  {image:<16} {name:<8} rel {rel:<7g} eps {rung['eps']:<11.4g} "
                    f"live {rung['relative_disagreement_live']:<10.3g} "
                    f"rtol {rung['required_rtol']:<10.3g} "
                    f"|grad| {rung['max_abs_gradient']:<10.3g}{note}",
                    flush=True,
                )
            # The best rung is chosen on the atol-free relative disagreement,
            # not on the checker's own verdict. Choosing on `required_rtol`
            # would hand the win to whichever step made the Jacobian row small
            # enough to slip under the absolute floor -- a perfect score for
            # having nothing to compare.
            # Rank steps on the scale-free number, and only over rungs that
            # actually had a live comparison to make. A rung where everything
            # sat inside the checker's absolute floor is a pass, but it carries
            # no relative information, so it cannot win the ranking.
            ranked = {rel: rung for rel, rung in ladder.items()
                      if rung["live_rows"] > 0}
            if ranked:
                best_rel = min(
                    ranked, key=lambda r: ranked[r]["relative_disagreement_live"]
                )
            else:
                clean = {rel: rung for rel, rung in ladder.items()
                         if rung["all_within_atol"]}
                if not clean:
                    raise SystemExit(
                        f"{image}/{name}: no rung produced either a live comparison "
                        f"or agreement within atol={ATOL:g}"
                    )
                best_rel = next(iter(clean))
            best = ladder[best_rel]
            component["input_paths"][name] = {
                "input_magnitude": magnitude,
                "by_rel_eps": ladder,
                "best_rel_eps": float(best_rel),
                "required_rtol": best["required_rtol"],
                "relative_disagreement": best["relative_disagreement_live"],
                "max_abs_gradient": best["max_abs_gradient"],
                "all_within_atol": best["all_within_atol"],
                # Summed over the whole ladder: a phantom row is structural, so
                # it shows up at every step and the count is the evidence.
                "phantom_rows": sum(r["phantom_rows"] for r in ladder.values()),
            }

        paths = component["input_paths"].values()
        component["required_rtol"] = max(p["required_rtol"] for p in paths)
        component["relative_disagreement"] = max(
            p["relative_disagreement"] for p in paths
        )
        component["phantom_rows"] = sum(p["phantom_rows"] for p in paths)
        record["components"][image] = component

    record["elapsed_seconds"] = round(time.time() - started, 1)
    record["total_comparisons"] = sum(
        entry["by_rel_eps"][repr(entry["best_rel_eps"])]["comparisons"]
        for component in record["components"].values()
        for entry in component["input_paths"].values()
    )
    worst = max(c["required_rtol"] for c in record["components"].values())
    worst_relative = max(
        c["relative_disagreement"] for c in record["components"].values()
    )
    phantom = sum(c["phantom_rows"] for c in record["components"].values())
    record["all_components_clean_at"] = worst
    record["worst_relative_disagreement"] = worst_relative
    record["phantom_rows_total"] = phantom
    record["cli_default_rtol"] = 0.1
    ok = bool(np.isfinite(worst_relative) and worst_relative < 1e-3)

    print()
    for image, component in record["components"].items():
        flag = (f"   PHANTOM SENSITIVITIES: {component['phantom_rows']}"
                if component["phantom_rows"] else "")
        print(
            f"  {image:<16} relative disagreement {component['relative_disagreement']:<10.3g} "
            f"(checker would need rtol {component['required_rtol']:.3g}){flag}"
        )
    if ok:
        print(
            f"\nPASS: every live comparison agrees to "
            f"{worst_relative:.3g} relative"
        )
    else:
        print("\nFAIL: a live comparison disagrees by more than 1e-3 relative")
    if phantom:
        print(
            f"\nNOTE: {phantom} comparisons reported a derivative with respect "
            f"to an input\n      the forward map does not read (the finite "
            f"difference was exactly zero).\n      That is a defect, not a "
            f"tolerance miss."
        )

    target = ROOT / "orchestrator" / "results" / "check_gradients.json"
    if N == AUDITED_N:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", newline="\n"
        )
        print(f"wrote {target}")
    else:
        print(f"grid {N} is not the audited {AUDITED_N}; {target.name} left untouched")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
