# Coldplate — Tesseract Hackathon 2026 submission

Coldplate composes a PyTorch material map, a C++/Eigen flow solver, and either
a JAX or independently written Fortran/Enzyme thermal solver in a two-way
buoyancy/advection fixed point. Newton–Krylov crosses the served-component
boundary with JVPs; a matrix-free implicit adjoint crosses it again with VJPs.

## Measured result

- Swapping JAX autodiff for compiler-differentiated Fortran changes the full
  coupled gradient by only **5.3 × 10⁻¹²**.
- The composed adjoint matches a true coupled finite difference to
  **8.3 × 10⁻⁶**, while a loop-cut shortcut is 86% wrong at the strong state.
- Under the same zero-sum raw-variable count and amplitude—not an equal
  realised-density budget—the composed sensitivity produces **58% more
  realised cooling** at the largest step.
- A retrospectively frozen 48-attempt extension retains every contrary result:
  **35 exact wins, 1 shortcut win, 3 ties, and 9 noncomparable attempts**—35/39
  among comparable cases, with an **81.1%** post-freeze descriptive
  seed-cluster-bootstrap lower endpoint.
- A separate frozen eight-step showdown is retained as incomplete: the
  composed step-six candidate did not converge, so there is no eight-step
  winner claim. Its common five-step prefix is labelled post-hoc only.

## Release integrity

The release contains the four-page technical paper; narrated demo, captions,
1920×1080 poster, and stream manifest; a deterministic source archive; exact
OCI image digests; checksummed commit/tag provenance; and a SHA-256 manifest.
The extended evidence manifest binds committed JSON bytes to their workflow
run, source commit, and GitHub artifact digest. The Enzyme LLVM-19 object and
its upstream licence are vendored and hash-checked, so a mutable nightly URL
cannot block or silently change the release build.

The nonlinear de Vahl Davis benchmark converges with all six observables within
1.2% of reference. The limitations are part of the result: this remains a
steady 2-D research prototype; the separate dimensional audit produced only 3
of 6 converged solves and temperatures outside its constant-property water
regime, so no fin-performance number is claimed; and the objective-aware
one-VJP diagnostic is not calibrated as a universal threshold for repelling
loops.

Quick judge path on Linux/amd64:

```bash
git clone https://github.com/TAUIL-Abd-Elilah/coldplate.git
cd coldplate
python3 -m pip install -r requirements-orchestrator.txt
bash scripts/judge_demo.sh --pull --grid 8
```

The `--pull` path uses the digest manifest attached to this release. It does not
log in to GHCR, so it also checks anonymous package visibility.
