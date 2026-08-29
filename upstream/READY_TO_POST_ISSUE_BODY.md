### Summary

Composed Tesseract pipelines often contain a coupled steady state `x = phi(x)`.
Differentiating it exactly means solving the adjoint system

```text
(I - dphi/dx)^T lambda = g
```

where `g` is the objective cotangent. The common shortcut is to cut the loop and
use `lambda_0 = g` — differentiate each component separately and ignore the
feedback. Sometimes that is fine and sometimes it is most of the gradient, and
today there is no cheap way for a user to tell which situation they are in.

The residual of that approximation is exactly `phi_x^T g`, which costs **one
VJP** — far less than the adjoint solve it would let you skip. I would like to
propose a small, dependency-free utility for computing it.

### Why is this needed?

The failure is silent. In the pipeline I built for the 2026 hackathon (a
two-way coupled cold plate composed across C++, JAX, Fortran/Enzyme and PyTorch
Tesseracts) the loop-cut gradient is 86% wrong with a third of the design
variables carrying the wrong sign, while the forward solution, the residual and
the convergence history all look healthy. An optimiser driven by it still
descends, which makes it easy to conclude the shortcut was fine.

The natural diagnostic — the spectral radius of the coupling Jacobian — is not
sufficient, because it is objective-blind. Holding one design and one operating
point fixed so `rho` cannot move, and changing only *what is being measured*, the
loop-cut error varied by a factor of 136. The normalised residual moves with it,
because it knows which direction the objective actually cares about.

I checked this off the physics as well: 2,377 random coupled fixed points across
four structural families, linear and nonlinear maps, everything closed form. The
log-correlation with the true error is +0.989, against +0.691 for the spectral
radius. It holds in every family. It also has a clear boundary, which I would
want documented rather than hidden: for repelling fixed points (`rho >= 1`) the
correlation collapses to +0.36, because the residual is amplified by
`(I - phi_x^T)^-1` and no contraction bound applies.

### Usage example

```python
from tesseract_jax import fixed_point_adjoint_residual

report = fixed_point_adjoint_residual(
    phi,                       # any JAX-traceable coupled map
    fixed_point=x_star,        # arbitrary PyTree
    objective_cotangent=g,     # dJ/dx at that state
)
report.gamma                   # ||phi_x^T g|| / ||g||, one VJP
report.repeated_relative_norms # successive applications, as diagnostics
```

Proposed signature:

```python
fixed_point_adjoint_residual(
    phi,
    fixed_point,
    objective_cotangent,
    num_repeats=4,
    check_fixed_point=False,
    thresholds=None,
    stability=None,
)
```

returning a frozen report with the relative residual, the raw cotangent and
residual norms, the repeated VJP norms, an optional primal fixed-point residual,
stability metadata, and a verdict **only when the caller supplies thresholds**.

### Design notes I would want reviewed

- **No default thresholds.** What counts as safe is application-specific. A
  library that shipped a number would be guessing on the user's behalf about the
  one thing that can hurt them, since a false `SAFE` tells someone to skip a
  computation they needed.
- **A known-repelling fixed point can never be labelled `SAFE`**, whatever the
  magnitude, for the reason above.
- **The repeated VJP norms are diagnostics, not a Neumann series.** They are
  useful because failure to decay is itself a warning, but calling them a
  convergent expansion would be wrong at a repelling fixed point.
- **PyTrees throughout**, so it matches how people actually hold coupled state.
- **The objective cotangent is explicit**, because mode alignment is exactly the
  information a spectral radius omits.

### Prior art and scope

The identity is standard — Padway and Mavriplis ([arXiv:2104.02826]) analyse
tangent and adjoint problems for fixed-point iterations, and relating a residual
to a solution error still requires the inverse coupled operator. Nothing here
claims otherwise. What I think is worth adding to `tesseract-jax` is that it
becomes one call against a composed pipeline, with the boundary documented.

### Would you want it?

Happy to open a PR with the module, unit tests covering arrays, nested PyTrees,
repeated VJPs, nonlinear local linearisation, zero cotangents, optional
fixed-point checking, caller thresholds and the repelling-map policy, plus API
docs and a short example — and to sign the CLA first. I am equally happy to be
told this belongs in user code rather than the library, or that it should ride
along with #154 instead of standing alone.

A working reference implementation and the measurements above are at
<https://github.com/TAUIL-Abd-Elilah/coldplate> (`fixed_point_adjoint.py`,
Apache-2.0), if it is useful to look at before deciding.
