# Releasing

Releases are built by GitHub Actions, not on a developer's machine. Pushing a tag that starts with `v` builds the launcher app and publishes it on the repository's Releases page.

## Steps

1. Make sure everything for the release is committed and pushed to `master`.
2. Tag it and push the tag:

       git tag -a v0.2.0 -m "Short summary of the release"
       git push origin v0.2.0

3. Open the repository's Actions tab and wait for the "build" run of that tag to finish, a few minutes.
4. Open Releases. The new release has `warband-metal-v0.2.0.zip` attached and release notes generated from the commits since the previous tag. Edit the notes if they need it.

The version in the zip's name and inside the app comes from the tag (`git describe`).

## What the workflow does

`.github/workflows/build.yml` runs on a macOS machine:

- on every push to `master` and on pull requests: `make dist`, with the zip kept as a build artifact of the run;
- on a `v*` tag: the same, and the zip is attached to a new release.

The game is not needed for the build. See "Build" in the README.

## Trying a release build locally

    make dist

produces `build/warband-metal-<version>.zip`, the same file the workflow makes. To check that a clean checkout builds, clone the repository into an empty folder and run it there.

## If something goes wrong

- **The run failed**: open it in the Actions tab; the failing step shows the compiler or download error. Fix, commit, push, then move the tag: `git tag -d v0.2.0`, `git push origin :refs/tags/v0.2.0`, and tag again.
- **A release should be withdrawn**: delete it on the Releases page, then delete the tag as above.

## The app is not signed

The app is not signed or notarised by Apple, so macOS blocks a downloaded copy until the user clears its quarantine flag (see "Using it" in the README). Signing needs a paid Apple Developer account; with one, a signing and notarisation step would go into the workflow after `make dist`.

## Cost

Actions is free for public repositories. While the repository is private, macOS machines use the free allowance ten times faster than Linux ones, and the workflow runs on every push to `master`.
