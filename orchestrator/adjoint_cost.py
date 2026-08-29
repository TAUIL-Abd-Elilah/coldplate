# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""What the composed adjoint costs, and what the alternatives would.

The repository argues at length that the *loop-cut* gradient is the wrong
shortcut. A reader is entitled to the other half of that argument: if you were
not going to build the coupled adjoint at all, what would you actually pay?

There are only two other ways to get a usable descent direction here.

  * **Finite differences over the design vector.** Exact-ish, needs no
    derivative from any component, and costs two coupled forward solves per
    design variable. At 48x48 that is 2 x 2304 = 4608 coupled solves for one
    gradient.
  * **The loop-cut shortcut**, which is cheap and is what the rest of this
    repository measures the error of.

This script measures the first honestly, and the wording matters. What it times
is not a re-solve of the *same* design: warm-started from its own answer, Newton
exits in a single iteration and you end up timing a convergence check, which
flatters finite differences by two orders of magnitude. What a differencing
sweep actually pays is a re-solve after perturbing one design variable, warm
started from the base state. That is what is timed here, several times, and the
median is taken.

The finite-difference figure is then `2 * n_design * t_probe` -- an
extrapolation from a measured per-probe time, labelled as one wherever it is
quoted. We do not run 4608 solves to prove a multiplication.

What is *not* claimed: that finite differences would fail. They would work.
They would take a very long time, and they would still be finite differences
through a solver whose own tolerance sets a noise floor -- which is why the
composed adjoint agreeing with them to 7.45e-6 is the accuracy ceiling of the
comparison rather than of the adjoint.

Usage:
    python orchestrator/adjoint_cost.py            # audited grid, records
    python orchestrator/adjoint_cost.py --N 16     # smoke, records nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

from pipeline import ColdPlate, Params

jax.config.update("jax_enable_x64", True)

#: The grid the README quotes. Only a run here rewrites the artefact.
AUDITED_N = 48

#: Committed optimisation histories, used for the whole-run arithmetic. These
#: are measurements from the actual optimisation, not re-timed here.
HISTORIES = {48: "history_composed_N48.json", 96: "history_composed_N96.json"}


def _timeit(fn, repeats: int):
    """Best-of-`repeats` wall clock, to blunt scheduler noise on a shared box."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.time()
        out = fn()
        best = min(best, time.time() - t0)
    return best, out


def measure(N: int, repeats: int = 3, probes: int = 5) -> dict:
    p = Params(Nx=N, Ny=N, Ra=3.0e4)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.25, 0.75, size=(N, N))

    with ColdPlate(params=p) as cp:
        mat = cp.material(rho)
        alpha, k = mat["alpha"], mat["k"]

        t_cold, (T_star, info) = _timeit(lambda: cp.solve_coupled(alpha, k), 1)
        if not info["ok"]:
            raise SystemExit(
                f"the coupled solve did not converge (residual {info['residual']:.3e})"
            )
        cold_newton = int(info["iters"])

        # What one finite-difference probe actually costs.
        #
        # Re-solving the *same* design from the converged state is not it: the
        # warm start is already the answer, Newton exits in one iteration, and
        # the timing is of a convergence check rather than a solve. A
        # differencing sweep perturbs one design variable and re-solves, so
        # that is what is timed here -- warm-started from the base state, the
        # way anyone running such a sweep would, with the warm start reset
        # before each probe so the probes are independent and alike.
        eps = 1e-4
        probe_seconds = []
        probe_newton = []
        probe_rng = np.random.default_rng(1)
        for _ in range(probes):
            perturbed = np.array(rho)
            i, j = probe_rng.integers(0, N, size=2)
            perturbed[i, j] += eps
            probe_mat = cp.material(perturbed)
            cp._T_warm = np.asarray(T_star)
            t0 = time.time()
            _, probe_info = cp.solve_coupled(probe_mat["alpha"], probe_mat["k"])
            probe_seconds.append(time.time() - t0)
            probe_newton.append(int(probe_info["iters"]))
            if not probe_info["ok"]:
                raise SystemExit("a finite-difference probe solve did not converge")
        # Round here, not on the way out: every figure derived below is
        # then exactly the product of the numbers the artefact publishes,
        # so a reader recomputing from them gets the printed answer.
        t_forward = round(float(np.median(probe_seconds)), 4)

        # The adjoint is timed under exactly the probe's conditions: a design
        # one step away from the base, warm-started from the base state. Timing
        # it on the base design instead would warm-start it at its own answer,
        # Newton would do no work, and the forward half of the gradient's cost
        # would vanish -- which would flatter the adjoint in a comparison whose
        # whole point is its cost.
        adjoint_seconds = []
        adjoint_rng = np.random.default_rng(2)
        used = {}
        res = None
        for _ in range(repeats):
            moved = np.array(rho)
            a, b = adjoint_rng.integers(0, N, size=2)
            moved[a, b] += eps
            cp._T_warm = np.asarray(T_star)
            before = dict(cp.stats)
            start = time.time()
            res = cp.value_and_grad(moved)
            adjoint_seconds.append(time.time() - start)
            used = {key: cp.stats[key] - before[key] for key in cp.stats}
        t_adjoint = round(float(np.median(adjoint_seconds)), 4)
        if not np.all(np.isfinite(np.asarray(res["grad"]))):
            raise SystemExit("the composed gradient is not finite; refusing to time it")

    n_design = N * N
    fd_solves = 2 * n_design
    record = {
        "N": int(N),
        "n_design_variables": int(n_design),
        "seconds_one_coupled_forward_solve": t_forward,
        "seconds_one_composed_gradient": t_adjoint,
        "finite_difference_probe": {
            "what_is_timed": (
                "re-solve after perturbing one design variable, warm-started "
                "from the base state, which is what a differencing sweep does"
            ),
            "probes": int(probes),
            "step": eps,
            "seconds": [round(value, 4) for value in probe_seconds],
            "newton_iterations": probe_newton,
        },
        "cold_solve": {
            "seconds": t_cold,
            "newton_iterations": cold_newton,
        },
        "cross_boundary_matvecs_per_gradient": {
            "jvp": int(used["jvp_matvecs"]),
            "vjp": int(used["vjp_matvecs"]),
        },
        "adjoint_timing": {
            "what_is_timed": (
                "value_and_grad on a design one step from the base, warm-started "
                "from the base state -- the same conditions as a probe"
            ),
            "seconds": [round(value, 4) for value in adjoint_seconds],
        },
        "central_difference_gradient": {
            "coupled_solves_required": int(fd_solves),
            "extrapolated_seconds": fd_solves * t_forward,
            "extrapolated_from": "measured seconds_one_coupled_forward_solve",
            # Timed as the median of perturbed, warm-started probe solves --
            # exactly what a differencing sweep pays per column, and the
            # cheapest figure we can honestly quote. The bias therefore runs
            # against our own conclusion.
            "timing_basis": (
                "median of perturbed warm-started probe solves, which is the "
                "cheapest honest per-solve figure and therefore favours finite "
                "differences"
            ),
        },
        "adjoint_speedup_over_central_differences": fd_solves * t_forward / t_adjoint,
    }

    history_path = (
        Path(__file__).resolve().parent / "results" / HISTORIES.get(N, "")
        if N in HISTORIES
        else None
    )
    if history_path is not None and history_path.exists():
        history = json.loads(history_path.read_text())
        run_seconds = sum(entry["seconds"] for entry in history)
        record["optimisation"] = {
            "source": history_path.name,
            "iterations": len(history),
            "measured_seconds": run_seconds,
            "J_start": history[0]["J"],
            "J_final": history[-1]["J"],
            "extrapolated_seconds_with_central_differences": (
                len(history) * fd_solves * t_forward
            ),
        }
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=AUDITED_N, help="grid size")
    ap.add_argument("--repeats", type=int, default=3, help="best-of-N timing")
    ap.add_argument("--probes", type=int, default=5,
                    help="perturbed warm-started solves timed for the FD cost")
    args = ap.parse_args(argv)

    print(f"timing the composed gradient at {args.N}x{args.N} ...")
    record = measure(args.N, args.repeats, args.probes)
    record["schema_version"] = 1
    record["repeats"] = int(args.repeats)

    fd = record["central_difference_gradient"]
    print(
        f"\n  design variables                 {record['n_design_variables']}\n"
        f"  one finite-difference probe      {record['seconds_one_coupled_forward_solve']:.3f} s"
        f"  (median of {record['finite_difference_probe']['probes']}, "
        f"{record['finite_difference_probe']['newton_iterations']} Newton iterations)\n"
        f"  one cold coupled solve           {record['cold_solve']['seconds']:.3f} s"
        f"  ({record['cold_solve']['newton_iterations']} Newton iterations)\n"
        f"  one composed gradient            {record['seconds_one_composed_gradient']:.3f} s"
        f"  ({record['cross_boundary_matvecs_per_gradient']['jvp']} JVP + "
        f"{record['cross_boundary_matvecs_per_gradient']['vjp']} VJP across the boundary)\n"
        f"  one central-difference gradient  {fd['coupled_solves_required']} coupled solves"
        f"  ~= {fd['extrapolated_seconds'] / 3600:.1f} h (extrapolated)\n"
        f"  ratio                            {record['adjoint_speedup_over_central_differences']:.0f}x"
    )
    if "optimisation" in record:
        opt = record["optimisation"]
        print(
            f"\n  the committed {opt['iterations']}-iteration optimisation took "
            f"{opt['measured_seconds'] / 60:.1f} min and reached J = {opt['J_final']:.4f};\n"
            f"  the same schedule on central differences extrapolates to "
            f"{opt['extrapolated_seconds_with_central_differences'] / 86400:.1f} days."
        )

    target = Path(__file__).resolve().parent / "results" / "adjoint_cost.json"
    if args.N == AUDITED_N:
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", newline="\n"
        )
        print(f"\nwrote {target}")
    else:
        print(f"\ngrid {args.N} is not the audited {AUDITED_N}; {target.name} left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
