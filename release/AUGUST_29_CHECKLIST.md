# August 29 publication sequence

The repository remains private during development. The workflow is hard-locked
until **2026-08-29 00:00 UTC**. The GitHub environment
`submission-production` already exists with a custom branch policy restricted
to `master`; its branch policy is the protection rule checked by the workflow.
After making the repository public, add a required reviewer with self-review
disabled only if an independent reviewer is available to approve both phases.
Do not add a reviewer gate to a solo submission: it would deadlock preparation
and publication. The existing `master` branch policy is already the protection
rule required by the workflow.

This checklist follows the [official hackathon page](https://pasteurlabs.ai/tesseract-hackathon-2026/)
and its [terms](https://pasteurlabs.ai/tesseract-hackathon-2026/terms_and_conditions.txt).

On August 29:

1. Revalidate both committed MP4s and their SRTs and manifests, the shared
   poster, the paper PDF, the provenance manifest, and the result JSON, then
   push the exact clean commit. Preparation refuses untracked or modified
   release bytes.

   Two narrations ship. `demo/coldplate_submission.mp4` is the canonical Edge
   render; `demo/coldplate_submission_local_voice.mp4` is the same film
   narrated locally, and it exists because the canonical audio's
   redistribution rights could not be confirmed. If a rights question is ever
   raised about the narration, the answer is to point at the local render, not
   to argue about the other one. `THIRD_PARTY_NOTICES.md` states the position
   for both; do not soften it.
2. Make the GitHub repository public and confirm its API is anonymously
   readable.
3. Run the `submission release` workflow with phase `prepare` and confirmation
   `PREPARE`. This tests the source, builds all four images, pushes commit and
   release tags, and creates a draft release with checksums. Record the prepared
   commit SHA from the workflow summary. The draft includes checksummed
   provenance tying the tag and source archive to that full SHA.
4. In GitHub Packages, independently set these four container packages to
   **Public**: `coldplate-stokes_brinkman`, `coldplate-thermal_advdiff`,
   `coldplate-thermal_fortran`, and `coldplate-material_map`. Repository
   visibility does not automatically change package visibility.
5. Run the same workflow with phase `publish` and confirmation `PUBLISH`.
   It uses an empty Docker credential directory to pull every recorded OCI
   digest, verifies every release checksum and actual video stream, checks out
   the prepared SHA (not the new dispatch SHA), then publishes only if the
   release is still a draft pinned to that commit.
6. Set the repository's public metadata, which judges see before any file:

   ```bash
   gh repo edit TAUIL-Abd-Elilah/coldplate \
     --description "A coupled cold-plate adjoint composed across C++/Eigen, JAX, Fortran+Enzyme and PyTorch Tesseracts. Tesseract Hackathon 2026, multi-physics track." \
     --homepage "https://tauil-abd-elilah.github.io/coldplate/docs/" \
     --add-topic tesseract --add-topic differentiable-simulation \
     --add-topic adjoint --add-topic topology-optimization \
     --add-topic multiphysics --add-topic enzyme-ad --add-topic jax
   ```

   Then enable Pages so `docs/index.html` has a public URL. Deploy from branch
   `master`, folder **`/` (root)** — not `/docs`: the page loads its figures
   from `../orchestrator/results/`, which only resolves when the whole
   repository is served. `.nojekyll` at the root keeps Jekyll out of the way.
   Upload `demo/poster.png` as the social preview in repository settings; that
   is the image every shared link renders.

7. Open a logged-out browser and recheck the repository, release, paper, video,
   hero assets, and the Pages URL. Run a credential-free clone on Linux/amd64.
8. Keep the repository, release, and packages public through judging. Complete
   the required LinkedIn post and official submission form before August 31.
   The form is at <https://tally.so/r/KYNZMg>; the LinkedIn post must carry the
   repository link and tag Pasteur Labs & ISI and Tesseract.

9. Optional, and only after the repository is public so the reference
   implementation is visible: file the `fixed_point_adjoint` feature request on
   `pasteurlabs/tesseract-jax` from
   `upstream/TESSERACT_JAX_PROPOSAL.md`. Their CONTRIBUTING asks for an issue
   before code and requires a signed CLA for a pull request, so the issue is the
   correct first step. Until a public URL exists, nothing anywhere in the
   submission may describe it as submitted or accepted.

Owner-only rule gates before the final form:

- confirm registration and the official eligibility declarations for every
  entrant: age/guardian consent, sanctions/residency, and no disqualifying
  Pasteur Labs employment, contract, or immediate-family relationship;
- confirm the work was created during the August 3–31 hackathon period, does
  not infringe third-party IP, and every member agrees to Apache-2.0;
- confirm one submission only, at most four people, and that every actual team
  member was identified at registration;
- confirm the Git author identities `Coldplate`, `pixgenx`, and
  `TAUIL-Abd-Elilah` are aliases of declared entrants, and deliberately accept
  the public history email rather than discovering it after publication;
- publish the required LinkedIn post with the repository link and required
  organization tags, then submit the official form exactly once.

Also acknowledge before submitting that the terms grant the host the stated
marketing/display licence and that a winner must provide applicable tax forms
within 30 days. These are owner declarations, not facts the repository can
prove.

Do not publish phase two when an anonymous digest pull fails. Fix package
visibility first; never replace a failed digest with a newly built image under
the same release record.

The workflow serializes prepare/publish runs without cancellation. If an old
draft under the requested tag has unexpected stale assets, preparation fails
instead of silently carrying them forward; inspect or remove that draft
manually before retrying.
