# Proposal: objective-aware fixed-point adjoint residual diagnostic

Target repository: `pasteurlabs/tesseract-jax`. Before posting, search the
current issue list and confirm whether this belongs under or alongside
[`#154`](https://github.com/pasteurlabs/tesseract-jax/issues/154); do not open a
duplicate without maintainer guidance.

## Ready-to-post issue

### Problem

Differentiable compositions often contain a coupled steady state `x = phi(x)`.
Cutting that loop and differentiating each component separately uses the
adjoint approximation `lambda_0 = g`, where `g` is the objective cotangent.
Users currently lack a small, reusable diagnostic for the residual of that
approximation.

For the exact adjoint equation

```text
(I - phi_x)^T lambda = g,
```

the loop-cut residual is exactly `phi_x^T g`.  Its relative L2 norm is an
objective-aware warning signal available from one VJP.  It is not a universal
adjoint-error bound, especially when the fixed point is repelling.

### Proposed API

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

The function accepts arbitrary JAX PyTrees and returns a frozen typed report
with `relative_residual` (`gamma` alias), raw cotangent/residual norms, repeated
raw and relative VJP norms, an optional primal fixed-point residual, stability
metadata, and an optional verdict.

No thresholds are supplied by the library: a verdict exists only when callers
provide thresholds calibrated on their own application.  Known repelling
stability vetoes a `SAFE` verdict because residual-to-error amplification is
then uncontrolled.  Repeated VJP norms remain diagnostics and are not described
as a convergent Neumann series.

### Why this API

- The objective cotangent is explicit because mode alignment is the useful
  information that a spectral radius alone omits.
- PyTree support matches JAX transformations and avoids flattening application
  state manually.
- Aggregate leaf L2 norms give one scale-independent number while the report
  retains raw norms for auditing.
- Fixed-point checking is opt-in, reusing the primal value already produced by
  `jax.vjp` without another evaluation of `phi`.
- Calibration and stability are explicit, preventing benchmark-specific
  policy from masquerading as a general guarantee.

## Ready-to-post PR plan

1. Add an isolated utility module containing `ResidualThresholds`,
   `FixedPointStability`, `FixedPointAdjointResidualReport`, and
   `fixed_point_adjoint_residual`.
2. Add focused unit tests covering arrays, nested PyTrees, repeated VJPs,
   nonlinear local linearization, zero cotangents, repeat validation, optional
   fixed-point checking, caller thresholds, and repelling-map policy.
3. Add API documentation explaining the adjoint residual identity, norm
   aggregation, the absence of default thresholds, and the distinction between
   residual and adjoint error.
4. Add a short example using a coupled map and an objective cotangent.
5. Keep existing coupling-specific APIs unchanged; discuss a compatibility
   wrapper only as a separate follow-up.

Suggested PR title: `Add objective-aware fixed-point adjoint residual diagnostic`

## CLA and contribution note

Before opening the PR, confirm the upstream repository's contribution guide,
sign the required CLA/DCO if prompted, and adapt copyright headers and naming
to the maintainers' conventions.  This document is a proposal only; nothing
has been submitted upstream.
