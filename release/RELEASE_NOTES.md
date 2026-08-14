# Coldplate — Tesseract Hackathon 2026 submission

Coldplate composes PyTorch, C++/Eigen, JAX, and Fortran/Enzyme Tesseracts in a
two-way buoyancy/advection fixed point and differentiates the converged system
with a matrix-free implicit adjoint.

The release contains the four-page technical paper, the narrated demo and
captions, a deterministic source archive, exact OCI image digests, and a
SHA-256 manifest.

Quick judge path on Linux/amd64:

```bash
git clone https://github.com/TAUIL-Abd-Elilah/coldplate.git
cd coldplate
python3 -m pip install -r requirements-orchestrator.txt
bash scripts/judge_demo.sh --pull --grid 8
```

The `--pull` path uses the digest manifest attached to this release. It does not
log in to GHCR, so it also checks anonymous package visibility.
