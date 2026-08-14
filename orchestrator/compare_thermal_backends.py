# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Swap a JAX component for a Fortran/Enzyme one and check nothing changes.

`thermal_advdiff` and `thermal_fortran` solve the same equation behind the same
schema, but obtain their derivatives by completely different means:

    thermal_advdiff   residual in JAX      -> jax.jvp / jax.vjp
    thermal_fortran   residual in Fortran  -> Enzyme LLVM pass

If Tesseract's contract means anything, these two must be interchangeable: the
composed pipeline should produce the same steady state *and the same end-to-end
gradient* with either one plugged in. That is what this script measures, at
three levels:

  1. component  T, and the JVP/VJP of each block, on identical inputs
  2. coupled    the converged fixed point of the full three-component loop
  3. gradient   dJ/drho_raw through the whole composition

Level 3 is the one that matters. It runs through the C++ fluid solver, the
PyTorch material map, and a thermal block that is in one case autodiff'd
Python and in the other compiler-differentiated Fortran.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from pipeline import ColdPlate, Params

jax.config.update("jax_enable_x64", True)


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def compare_components(N: int, rng) -> float:
    """Level 1: the two thermal blocks alone, on identical inputs."""
    u = jnp.asarray(rng.normal(size=(N, N + 1)) * 5.0).at[:, 0].set(0.0).at[:, N].set(0.0)
    v = jnp.asarray(rng.normal(size=(N + 1, N)) * 5.0).at[0, :].set(0.0).at[N, :].set(0.0)
    k = jnp.asarray(rng.uniform(0.02, 1.0, size=(N, N)))
    inputs = {"u": u, "v": v, "k": k, "q_chip": 1.0, "chip_frac": 0.4}

    worst = 0.0
    served = {}
    try:
        for name in ("thermal_advdiff", "thermal_fortran"):
            t = Tesseract.from_image(name)
            t.serve()
            served[name] = t

        jx, fo = served["thermal_advdiff"], served["thermal_fortran"]
        T_jax = apply_tesseract(jx, inputs)["T"]
        T_for = apply_tesseract(fo, inputs)["T"]
        e = relerr(T_for, T_jax)
        worst = max(worst, e)
        print(f"[1 component] forward T          rel err {e:.3e}")
        print(f"              max|T| jax {float(jnp.abs(T_jax).max()):.6f}  "
              f"fortran {float(jnp.abs(T_for).max()):.6f}")

        # tangents and cotangents through each block
        du = jnp.asarray(rng.normal(size=u.shape))
        dv = jnp.asarray(rng.normal(size=v.shape))
        dk = jnp.asarray(rng.normal(size=k.shape))

        def block(t, uu, vv, kk):
            return apply_tesseract(t, {**inputs, "u": uu, "v": vv, "k": kk})["T"]

        _, jvp_jax = jax.jvp(lambda a, b, c: block(jx, a, b, c), (u, v, k), (du, dv, dk))
        _, jvp_for = jax.jvp(lambda a, b, c: block(fo, a, b, c), (u, v, k), (du, dv, dk))
        e = relerr(jvp_for, jvp_jax)
        worst = max(worst, e)
        print(f"[1 component] JVP                rel err {e:.3e}")

        Tbar = jnp.asarray(rng.normal(size=(N, N)))
        _, vjp_jax = jax.vjp(lambda a, b, c: block(jx, a, b, c), u, v, k)
        _, vjp_for = jax.vjp(lambda a, b, c: block(fo, a, b, c), u, v, k)
        gj = vjp_jax(Tbar)
        gf = vjp_for(Tbar)
        e = max(relerr(gf[i], gj[i]) for i in range(3))
        worst = max(worst, e)
        print(f"[1 component] VJP (u, v, k)      rel err {e:.3e}")

        # Repeat the component contract at the de Vahl Davis wall mode. A
        # deliberately nonzero q_chip catches an accidental fall-through to
        # the cold-plate boundary condition: mode 2 must ignore that flux.
        cavity_inputs = {
            **inputs,
            "q_chip": 13.0,
            "bc_mode": 2.0,
            "t_hot": 1.0,
        }

        def cavity_block(t, uu, vv, kk):
            return apply_tesseract(
                t, {**cavity_inputs, "u": uu, "v": vv, "k": kk}
            )["T"]

        Tc_jax = cavity_block(jx, u, v, k)
        Tc_for = cavity_block(fo, u, v, k)
        e = relerr(Tc_for, Tc_jax)
        worst = max(worst, e)
        print(f"[1 cavity  ] forward T          rel err {e:.3e}")

        _, jvp_jax = jax.jvp(
            lambda a, b, c: cavity_block(jx, a, b, c),
            (u, v, k), (du, dv, dk),
        )
        _, jvp_for = jax.jvp(
            lambda a, b, c: cavity_block(fo, a, b, c),
            (u, v, k), (du, dv, dk),
        )
        e = relerr(jvp_for, jvp_jax)
        worst = max(worst, e)
        print(f"[1 cavity  ] JVP                rel err {e:.3e}")

        _, vjp_jax = jax.vjp(lambda a, b, c: cavity_block(jx, a, b, c), u, v, k)
        _, vjp_for = jax.vjp(lambda a, b, c: cavity_block(fo, a, b, c), u, v, k)
        gj, gf = vjp_jax(Tbar), vjp_for(Tbar)
        e = max(relerr(gf[i], gj[i]) for i in range(3))
        worst = max(worst, e)
        print(f"[1 cavity  ] VJP (u, v, k)      rel err {e:.3e}")
    finally:
        for t in served.values():
            try:
                t.teardown()
            except Exception:  # noqa: BLE001
                pass
    return worst


def compare_pipeline(N: int, Ra: float = 1.0e4) -> float:
    """Levels 2 and 3: the coupled state and the end-to-end gradient.

    Uses its own generator and a Rayleigh number where this design's fixed
    point converges cleanly. The comparison would be valid either way -- both
    backends do the same thing to the same iterate -- but a claim about the
    end-to-end gradient is only worth making at a converged steady state.
    """
    rho = np.random.default_rng(0).uniform(0.25, 0.75, size=(N, N))
    p = Params(Nx=N, Ny=N, Ra=Ra)
    out = {}

    for backend in ("thermal_advdiff", "thermal_fortran"):
        images = {
            "material": "material_map",
            "fluid": "stokes_brinkman",
            "thermal": backend,
        }
        t0 = time.time()
        with ColdPlate(params=p, images=images) as cp:
            res = cp.value_and_grad(rho)
        out[backend] = res
        print(f"[2/3 pipeline] {backend:16s} J = {res['J']:.10f}  "
              f"fixed point {res['info']['iters']} Newton iters, "
              f"residual {res['info']['residual']:.1e} "
              f"{'converged' if res['info']['ok'] else 'NOT CONVERGED'}"
              f"  ({time.time()-t0:.0f}s)")
        if not res["info"]["ok"]:
            print("   warning: comparing at a non-converged state; "
                  "lower Ra for a clean comparison")

    a, b = out["thermal_advdiff"], out["thermal_fortran"]
    e_T = relerr(b["T"], a["T"])
    e_g = relerr(b["grad"], a["grad"])
    ga, gb = a["grad"].ravel(), b["grad"].ravel()
    cos = float(ga @ gb / (np.linalg.norm(ga) * np.linalg.norm(gb)))

    print(f"\n[2 coupled ] converged T*         rel err {e_T:.3e}")
    print(f"[3 gradient] dJ/drho end-to-end   rel err {e_g:.3e}")
    print(f"[3 gradient] cosine between them  {cos:.12f}")
    print(f"[3 gradient] |J difference|       {abs(a['J']-b['J']):.3e}")
    return max(e_T, e_g)


def main(N: int = 16) -> int:
    rng = np.random.default_rng(0)
    print(f"grid {N}x{N}\n")
    print("Swapping the thermal block: JAX autodiff  <->  Fortran + Enzyme compiler AD\n")

    w1 = compare_components(N, rng)
    print()
    w2 = compare_pipeline(N)

    worst = max(w1, w2)
    print(f"\nworst rel err across all three levels: {worst:.3e}")
    ok = worst < 1e-8
    print("PASS: the two backends are interchangeable" if ok else "FAIL: backends disagree")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 16))
