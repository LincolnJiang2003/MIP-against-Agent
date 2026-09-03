# Third-party notices

Project-authored code is licensed under the Apache License 2.0. This repository
also contains adapted or redistributed components from the following projects.
Their licenses remain in force and are not replaced by the top-level license.

| Component | Location | Upstream | License |
|---|---|---|---|
| Windows Agent Arena agent, SOM, and prompt glue | `src/mip_attack/agent.py`, `a11y.py`, `parsing/utils`, `prompts.py` | <https://github.com/microsoft/WindowsAgentArena> | MIT; see `third_party/licenses/WINDOWS-AGENT-ARENA-MIT.txt` |
| Windows Agent Arena / OSWorld task fixtures | `data/examples`, `data/splits` | <https://github.com/microsoft/WindowsAgentArena>, <https://github.com/xlang-ai/OSWorld> | MIT / Apache-2.0; see bundled license texts |
| GroundingDINO source | `src/mip_attack/parsing/groundingdino` | <https://github.com/IDEA-Research/GroundingDINO> | Apache-2.0; see `third_party/licenses/GROUNDINGDINO-APACHE-2.0.txt` |
| OmniParser adapter and annotation utilities | `src/mip_attack/parsing/omniparser` | <https://github.com/microsoft/OmniParser> | CC BY 4.0; see `third_party/licenses/OMNIPARSER-CC-BY-4.0.txt` |
| Legacy OmniParser detector weights | downloaded by `scripts/prepare_models.py` | <https://huggingface.co/microsoft/OmniParser> | AGPL-3.0; the download includes its license |
| Florence icon-caption weights | downloaded by `scripts/prepare_models.py` | <https://huggingface.co/microsoft/OmniParser> | MIT; the download includes its license |
| Arial 2.82 rendering font | supplied locally by the user; never redistributed | <https://learn.microsoft.com/en-gb/typography/font-list/arial> | Proprietary Monotype/Microsoft font; obtain and use under an applicable license |

The 25 desktop screen PNGs tracked in `data/screens/desktop/` are research
artifacts and are not covered by the repository's code license. Social-media
screens are not redistributed. See `data/README.md` for the asset boundary.
