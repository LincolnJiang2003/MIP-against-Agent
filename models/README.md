# Model artifacts

Run `python scripts/prepare_models.py --arial-font /path/to/arial.ttf` from the
repository root. Model files are downloaded here and are intentionally excluded
from Git. The script pins and verifies the exact parser and rendering artifacts
used by the paper code.

The original experiment bundled proprietary Arial 2.82. It is not redistributed
in this repository. Supply a legally obtained `arial.ttf` with SHA-256
`35c0f3559d8db569e36c31095b8a60d441643d95f59139de40e23fada819b833`.
Microsoft lists Arial as exclusive to the products that supply it and provides
separate redistribution guidance: <https://learn.microsoft.com/en-gb/typography/font-list/arial>.

The legacy OmniParser detector checkpoint is a PyTorch pickle and is AGPL-3.0.
Only load the verified checksum listed in `manifest.json`; GroundingDINO is
loaded in weights-only mode. Llama 3.2 is gated; accept Meta's license and
authenticate with Hugging Face before crafting.
