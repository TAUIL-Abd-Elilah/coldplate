## The problem

`check_gradients` takes one `eps` and applies it, unscaled, to every
differentiated input. Its docstring says otherwise:

> `eps`: The epsilon to use for finite differences, **as a fraction of the maximum absolute value of each input**.

So a reader picks a relative step and gets an absolute one. This is
[#706](https://github.com/pasteurlabs/tesseract-core/issues/706).

For inputs of order one the difference never surfaces. For an input carrying
physical scale it decides whether the check means anything. In the pipeline
that prompted this, a Brinkman drag coefficient sits at ~2 × 10⁴ at the state
being checked, next to a temperature field of order 1. Reading the docstring, I
chose `--eps=1e-6` intending a step of about 0.02 on the drag; what was taken
was 10⁻⁶, a relative step of 5 × 10⁻¹¹. The central difference then dissolved
into the solver's own convergence noise and the checker reported **400 failures
out of 400 checks on both gradient endpoints**, against derivatives that agree
with an independent JAX reimplementation to better than 1e-9 and satisfy the
adjoint identity to the same.

The failure mode is the bad kind: it accuses your derivative, confidently, and
nothing in the output suggests the step was the problem. Holding everything else
fixed and changing only the step on that one input takes 120/120 failures to
4/120.

## The change

`eps` also accepts `dict[str, float]`, one step size per differentiated input
path:

```python
check_gradients(
    api_module, {"inputs": inputs},
    input_paths=["alpha", "T"],
    eps={"alpha": 2e-2, "T": 3e-6},
)
```

A scalar behaves exactly as before — same code path, same arithmetic — and a
mapping whose values are all equal gives identical results to that scalar, which
is tested. A mapping must name every path being checked and no others, so a typo
is a `ValueError` rather than a silently skipped input.

The docstring now describes what the code does. The CLI `--eps` help says the
step is absolute and points at `--input-paths` for inputs of differing scale;
the CLI itself stays scalar, since the per-path form is a library concern.

## Also: `--show-progress` had no off switch

It was declared with only its positive name, so `--no-show-progress` did not
exist and the bar could not be turned off. Scripted and CI use is exactly where
this command is most valuable, and where the captured output is otherwise a wall
of progress-bar escape codes. Now declared as
`--show-progress/--no-show-progress`.

## Validation

- `tests/runtime_tests/test_finite_differences.py`: scalar/mapping equivalence;
  that a per-path step actually reaches the difference; and both validation
  errors.
- The equivalence test is the one that matters for backwards compatibility. The
  "reaches the difference" test uses a cubic, because a central difference of
  one is wrong by exactly `h²` while the analytic Jacobian is exact — so a step
  large enough on one input fails that input's check while the other keeps
  passing, which cannot happen if the mapping is ignored. Its two inputs feed
  separate outputs deliberately: summed into one, a perturbation of the small
  input vanishes below float64 resolution against the large one's magnitude and
  the difference comes back exactly zero, which is a real effect but not the one
  under test.
- `tests/runtime_tests/` passes (422 tests), and `pre-commit` is clean on the
  changed files.

## Relation to other work

[#516](https://github.com/pasteurlabs/tesseract-core/issues/516) asks for
automated normalization strategies for finite differences; this is the narrower
half — the checker's documented behaviour made real today, with the choice left
explicit rather than inferred.
[#712](https://github.com/pasteurlabs/tesseract-core/pull/712) does the
corresponding thing for `runtime/experimental/finite_differences.py`, the
helpers that *make* a Tesseract differentiable. This touches
`runtime/testing/finite_differences.py`, the checker; the two do not overlap in
file or in API, and the argument shape here deliberately matches that one so
users meet the same convention in both places.

Happy to adjust the validation strictness, the error wording, or to split the
`--show-progress` fix into its own PR if you would rather keep them separate.

---

Found while composing four Tesseracts — C++/Eigen, JAX, Fortran/Enzyme and
PyTorch — into a two-way coupled adjoint for the 2026 hackathon.
