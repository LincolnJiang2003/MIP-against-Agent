# Experiment data

This directory contains the exact 12/12 train/evaluation prompt split. The 25
desktop PNGs are tracked directly under `data/screens/desktop/`. Their
checksums can be verified with:

```bash
python scripts/verify_assets.py
```

The prompt fixtures derive from Windows Agent Arena and OSWorld; see
`THIRD_PARTY.md`. The desktop screens are research artifacts and are not
covered by the repository's code license.

The social-media screenshots used in the paper are not redistributed. They
contain third-party public posts, account identifiers, profile images, and
post media, and public availability alone does not establish redistribution
permission. A blurred substitute would not be the same experimental input.
The Bluesky configs are retained to document the paper protocol, but require
independently obtained screenshots that you are authorized to use at their
configured paths.
