# August 29 publication sequence

The repository remains private during development. On August 29:

1. Make the GitHub repository public and confirm its API is anonymously
   readable.
2. Run the `submission release` workflow with phase `prepare` and confirmation
   `PREPARE`. This tests the source, builds all four images, pushes commit and
   release tags, and creates a draft release with checksums.
3. In GitHub Packages, independently set these four container packages to
   **Public**: `coldplate-stokes_brinkman`, `coldplate-thermal_advdiff`,
   `coldplate-thermal_fortran`, and `coldplate-material_map`. Repository
   visibility does not automatically change package visibility.
4. Run the same workflow with phase `publish` and confirmation `PUBLISH`.
   It uses an empty Docker credential directory to pull every recorded OCI
   digest, verifies all release checksums, then publishes the draft.
5. Open a logged-out browser and recheck the repository, release, paper, video,
   and hero assets. Run a credential-free clone on Linux/amd64.
6. Keep the repository, release, and packages public through judging. Complete
   the required LinkedIn post and official submission form before August 31.

Do not publish phase two when an anonymous digest pull fails. Fix package
visibility first; never replace a failed digest with a newly built image under
the same release record.
