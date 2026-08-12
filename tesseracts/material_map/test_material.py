# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the PyTorch material map and its derivatives against finite differences."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tesseract_api as api  # noqa: E402


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main():
    N = 10
    rng = np.random.default_rng(11)
    rho = rng.uniform(0.1, 0.9, size=(N, N))
    inputs = api.InputSchema(rho_raw=rho, filter_radius=2.0, beta=4.0, penal=3.0)

    out = api.apply(inputs)
    print(f"[apply] rho_phys range [{out.rho_phys.min():.4f}, {out.rho_phys.max():.4f}]")
    print(f"[apply] k range        [{out.k.min():.4f}, {out.k.max():.4f}]")
    print(f"[apply] alpha range    [{out.alpha.min():.3e}, {out.alpha.max():.3e}]")

    # The filter must be unbiased at the walls: a uniform field has to stay
    # spatially uniform, or the border would be darkened by the missing
    # neighbours. (Its *value* is allowed to move, because the Heaviside
    # projection is deliberately not the identity for beta > 0.)
    flat = api.apply(api.InputSchema(rho_raw=np.full((N, N), 0.42), filter_radius=2.0, beta=1.0))
    print(f"[filter] uniform in -> spatial std out = {flat.rho_phys.std():.3e}  (want ~0)")

    # The projection fixes rho_f = eta to exactly 0.5, for any beta.
    at_eta = api.apply(
        api.InputSchema(rho_raw=np.full((N, N), 0.5), filter_radius=2.0, beta=8.0, eta=0.5)
    )
    print(f"[project] rho_f=eta -> rho_phys dev from 0.5 = "
          f"{np.max(np.abs(at_eta.rho_phys - 0.5)):.3e}")

    # ---- VJP vs central finite differences ----
    cot = {n: rng.normal(size=(N, N)) for n in ("k", "alpha", "rho_phys")}
    vjp = api.vector_jacobian_product(inputs, {"rho_raw"}, set(cot), cot)["rho_raw"]

    def scalar(r):
        o = api.apply(api.InputSchema(rho_raw=r, filter_radius=2.0, beta=4.0, penal=3.0))
        return sum(float((getattr(o, n) * cot[n]).sum()) for n in cot)

    eps = 1e-6
    idx = [(int(a), int(b)) for a, b in rng.integers(0, N, size=(6, 2))]
    fd = []
    for j, i in idx:
        rp, rm = rho.copy(), rho.copy()
        rp[j, i] += eps
        rm[j, i] -= eps
        fd.append((scalar(rp) - scalar(rm)) / (2 * eps))
    got = np.array([vjp[j, i] for j, i in idx])
    print(f"[vjp   ] vs finite-diff rel err = {relerr(got, np.array(fd)):.3e}")

    # ---- JVP / VJP adjoint consistency ----
    tan = rng.normal(size=(N, N))
    jvp = api.jacobian_vector_product(inputs, {"rho_raw"}, set(cot), {"rho_raw": tan})
    lhs = sum(float((jvp[n] * cot[n]).sum()) for n in cot)
    rhs = float((tan * vjp).sum())
    mismatch = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    print(f"[adjoint] <J dx, ybar> = {lhs:.12e}")
    print(f"[adjoint] <dx, J^T yb> = {rhs:.12e}")
    print(f"[adjoint] rel mismatch = {mismatch:.3e}")

    worst = max(relerr(got, np.array(fd)), mismatch)
    print(f"\nworst rel err: {worst:.3e}")
    print("PASS" if worst < 1e-6 else "FAIL")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
