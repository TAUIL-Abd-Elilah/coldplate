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

1. Revalidate the committed MP4, SRT, poster, video manifest, paper PDF,
   provenance manifest, and result JSON, then push the exact clean commit.
   Preparation refuses untracked or modified release bytes.
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
6. Open a logged-out browser and recheck the repository, release, paper, video,
   and hero assets. Run a credential-free clone on Linux/amd64.
7. Keep the repository, release, and packages public through judging. Complete
   the required LinkedIn post and official submission form before August 31.

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
