# Contributing to RoboVoLo

Thank you for your interest in contributing.

## Developer Certificate of Origin

RoboVoLo accepts contributions under the Developer Certificate of Origin (DCO), version 1.1. By signing off a commit, you certify that you have the right to submit the contribution under this repository's license.

Read the DCO at <https://developercertificate.org/>.

Sign off every commit with:

```bash
$ git commit -s -m "Add cool feature."
```

This adds a line in the following form to the commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and an email address you control. Unsigned commits are not accepted. If necessary, add a sign-off to your latest commit with `git commit --amend --signoff`; preserve authorship when amending another contributor's work.

### Full text of the DCO

```text
Developer Certificate of Origin
Version 1.1
Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Licensing

By contributing, you agree that your contribution will be licensed under the Apache License 2.0 unless an authorized maintainer explicitly approves another license for a third-party component.

Do not submit code, assets, textures, scenes, or other material unless you have the right to do so. Preserve existing third-party notices exactly. New third-party material must include:

- the upstream name and source URL;
- the copyright holder;
- the exact license and required attribution;
- whether the material was modified; and
- corresponding updates to `THIRD_PARTY_NOTICES.md` and, for assets, `ASSET_LICENSES.md`.

NVIDIA-authored Python files must use the repository's Apache-2.0 header. Do not add NVIDIA SimReady or other Omniverse asset payloads to Git or Git LFS; those materials are obtained separately under their own terms.

## Submitting changes

1. Fork the repository and create a focused branch.
2. Make the change and update documentation and manifests as needed.
3. Run the relevant Python syntax and installer checks.
4. Ensure Git LFS is installed when changing tracked asset files.
5. Commit with `git commit -s`.
6. Open a pull request describing the change, its validation, and the provenance of any new material.
