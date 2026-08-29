# Ready-to-post feature request

`pasteurlabs/tesseract-jax` CONTRIBUTING asks for an issue before code and
requires a signed CLA for a pull request, so an issue is the correct first step.
Nothing here has been posted. Do not describe it anywhere as submitted or
accepted until a public URL exists.

Before posting, re-read the current issue list and decide whether this belongs
under or alongside [#154](https://github.com/pasteurlabs/tesseract-jax/issues/154).

Post it with:

```bash
gh issue create --repo pasteurlabs/tesseract-jax \
  --title "Feature request: objective-aware residual diagnostic for fixed-point adjoints" \
  --body-file upstream/READY_TO_POST_ISSUE_BODY.md
```

The body is kept in a separate file so it can be posted verbatim.

## Why this is worth their time, in one line

Anyone composing Tesseracts into a coupled steady state has to decide whether to
build the coupled adjoint or differentiate the components separately, and there
is currently no cheap way to tell. The residual of the shortcut is one VJP, and
it predicts the damage.

## What the issue does *not* do

- it does not ask maintainers to write anything;
- it does not propose default thresholds, because the safe boundary is
  application-specific and a library that guessed would be worse than useless;
- it does not claim the analysis is novel — the identity is standard. The
  contribution is making it one call, and measuring what it predicts.
