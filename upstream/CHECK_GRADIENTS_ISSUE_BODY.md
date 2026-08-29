### Description

`tesseract-runtime check-gradients` documents `--eps` as a *relative* step:

> `eps`: The epsilon to use for finite differences, **as a fraction of the maximum absolute value of each input**.
>
> — `tesseract_core.runtime.testing.finite_differences.check_gradients`, [`finite_differences.py#L579`](https://github.com/pasteurlabs/tesseract-core/blob/main/tesseract_core/runtime/testing/finite_differences.py#L579)

It is applied as an *absolute* step. `_perturb_input` does `input_val[idx] += eps`
([L178](https://github.com/pasteurlabs/tesseract-core/blob/main/tesseract_core/runtime/testing/finite_differences.py#L178)),
and nothing between the CLI and that line scales `eps` by the input's magnitude.

For an input whose values are O(1) the difference never shows up. For an input
with a large scale it is the difference between a working check and a confident
false alarm, because the failure is silent and points at the wrong thing: the
tool reports that your *derivative* is wrong, when what actually happened is
that the central difference was taken at a step 5 orders of magnitude smaller
than the one you asked for, and dissolved into your solver's own convergence
noise.

That is not hypothetical. I hit it on a Stokes–Brinkman solver whose two
differentiable inputs are a Brinkman drag coefficient `alpha` spanning
`[5.9e3, 2.0e4]` at the state being checked and a temperature field `T` spanning
`[0.05, 2.9]`. Reading the docstring, I chose `--eps=1e-6` as a relative step —
intending a step of about `2e-2` on `alpha`. What was actually taken was `1e-6`,
a relative step of 5e-11, and the result was
**400 failures / 400 checks on both `jacobian_vector_product` and
`vector_jacobian_product`** — a 100% failure rate on derivatives that agree with
an independent JAX reimplementation to better than 1e-9, and that satisfy the
adjoint identity `<J d, w> == <d, Jᵀ w>` to the same tolerance (both asserted by
the project's own component tests).

There is a second consequence of `eps` being a single global absolute number,
which I mention because the fix for one may inform the other: a Tesseract whose
differentiable inputs have very different magnitudes has **no single `eps` that
works for all of them**. A step suited to `alpha ~ 2e4` is enormous for
`T ~ 3`, and vice versa. The workaround is to invoke the checker once per
`--input-paths` with a hand-scaled `--eps`, which is what I ended up doing, but
it is only discoverable after the tool has already told you your gradients are
broken.

### Steps to reproduce

No container needed — the mismatch is visible directly:

```python
import numpy as np
from tesseract_core.runtime.testing.finite_differences import _perturb_input

inputs = {"x": np.array([1e5, 2e5])}
eps = 1e-6

actual = _perturb_input(inputs, "x", (0,), eps)["x"][0] - inputs["x"][0]
documented = eps * np.abs(inputs["x"]).max()

print(f"documented step (eps * max|x|) = {documented:g}")   # 0.2
print(f"actual step taken              = {actual:g}")       # 1e-06
print(f"ratio                          = {documented/actual:.0f}x")  # 200001x
```

End to end, against a served Tesseract with a large-scale input:

```bash
tesseract run stokes_brinkman check-gradients @payload.json \
    --runtime-args '--eps=1e-6 --rtol=1e-6 --seed=7 --max-evals=200'
```

### Logs

```bash
documented step (eps * max|x|) = 0.2
actual step taken              = 9.99993e-07
ratio                          = 200001x

⚠️ Gradient check for jacobian_vector_product failed ⚠️ (400 failures / 400 checks)
⚠️ Gradient check for vector_jacobian_product failed ⚠️ (400 failures / 400 checks)
```

Same component, same seed, same `rtol=1e-6`, same 120 comparisons per endpoint.
The only thing that changes across these three rows is the step, and the middle
row is the CLI default:

```bash
  alpha  eps=2.03e-2  (rel 1e-6)   jvp   4/120 fail   vjp   4/120 fail
  alpha  eps=1e-4     (rel 4.9e-9) jvp 120/120 fail   vjp 120/120 fail
  alpha  eps=1e-6     (rel 4.9e-11) jvp 120/120 fail  vjp 120/120 fail
```

The top row is `--eps` scaled per input path the way the docstring describes.
That is the workaround, not a fix.

### Relation to existing issues

[#516](https://github.com/pasteurlabs/tesseract-core/issues/516) proposes
automated normalization strategies for finite differences. This report is the
narrower, and I think more urgent, half of that: the docstring already tells
users normalization is in place, so someone reading it does not know they need
#516 — they think they have it, and the tool contradicts them without saying
why. Whatever shape #516 eventually takes, the gap between what `check-gradients`
documents today and what it does is worth closing on its own.

[#438](https://github.com/pasteurlabs/tesseract-core/issues/438) made these
arguments reachable from `tesseract run` via `--runtime-args`, which is how I
was able to set `--eps` at all; the `--show-progress` note below is a leftover
from that surface.

### Possible fixes

Either direction resolves it; they are not equivalent for users:

1. **Implement the documented behaviour** — scale `eps` per input path by
   `max(abs(value))` (guarding the all-zero case) inside
   `check_endpoint_gradients`, so one `--eps` is meaningful across inputs of
   different magnitudes. This is the behaviour the docstring already promises,
   and it makes the default `1e-4` sensible for a much wider class of
   Tesseracts.
2. **Correct the docstring** to say `eps` is an absolute step, and say in the
   CLI help that inputs of differing scale need one invocation per
   `--input-paths` with an appropriate `--eps`.

I would prefer (1), because the current default silently under-resolves any
input larger than about `1e2` and the failure mode is a false accusation
against the user's own code. But (2) is a one-line change and removes the trap.

### A smaller thing, in the same command

`--show-progress` is declared with only its positive name and a default of
`True`
([`cli.py#L362`](https://github.com/pasteurlabs/tesseract-core/blob/main/tesseract_core/runtime/cli.py#L362)),
so there is no way to turn the progress bar off:

```console
$ tesseract run material_map check-gradients @payload.json \
      --runtime-args '--eps=1e-6 --no-show-progress'
No such option: --no-show-progress (Possible options: --show-progress)
```

Scripted and CI use is exactly where this command is most valuable, and that is
also where the captured output is a wall of progress-bar escape codes.
`typer.Option("--show-progress/--no-show-progress")` would fix it.

### OS

Linux

### Tesseract version

`1.11.0` (behaviour confirmed unchanged on `main` at the time of writing:
docstring at `finite_differences.py#L579`, absolute step at `#L178`,
`--show-progress` at `cli.py#L362`).

---

Found while composing four Tesseracts into a coupled cold-plate adjoint for the
2026 hackathon; the harness that works around it is
[`orchestrator/check_gradients.py`](https://github.com/TAUIL-Abd-Elilah/coldplate/blob/master/orchestrator/check_gradients.py)
in <https://github.com/TAUIL-Abd-Elilah/coldplate>. Happy to open a PR for
either fix, and to sign the CLA first.
