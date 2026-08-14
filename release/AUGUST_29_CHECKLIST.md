# August 29 publication sequence

The repository remains private during development. The workflow is hard-locked
until **2026-08-29 00:00 UTC**. Before that date, create the GitHub environment
`submission-production`, give it at least one protection rule (preferably a
required reviewer with self-review disabled), and restrict deployment branches
to the submission branch.

On August 29:

1. Render the final video and evidence, commit the MP4, SRT, poster, video
   manifest, paper PDF, and result JSON, then push that exact clean commit.
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

Do not publish phase two when an anonymous digest pull fails. Fix package
visibility first; never replace a failed digest with a newly built image under
the same release record.

The workflow serializes prepare/publish runs without cancellation. If an old
draft under the requested tag has unexpected stale assets, preparation fails
instead of silently carrying them forward; inspect or remove that draft
manually before retrying.
