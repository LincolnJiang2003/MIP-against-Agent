# MIP against Agent

Reference implementation for [MIP against Agent: Malicious Image Patches
Hijacking Multimodal OS Agents](https://papers.neurips.cc/paper_files/paper/2025/hash/1add4b4f8373c64504cfe4a3054700a1-Abstract-Conference.html). It crafts
8-bit RGB image patches against Llama 3.2 Vision–based OS agents and evaluates
transfer across prompts, screens, parsers, and VLMs.

This is offensive-security research code. It produces patches intended to
hijack an OS agent's action generation. Use it only in an isolated research
environment with systems and models you are authorized to test. The code 
intentionally does not execute generated actions.

## Included experiments

The public desktop assets support two directly executable protocols covering
every desktop paper evaluation except execution-step/multi-turn transfer:

| Protocol | Crafting inputs | Evaluation inputs |
| --- | --- | --- |
| Targeted | One seeded sample from `P+ × S+` | `(p,s)`, `P- × {s}`, and `{p} × S-` |
| Universal | All `P+ × S+` inputs and configured crafting parsers | Every train/test prompt × train/test screen × evaluation-parser combination |

The targeted pair is sampled uniformly from the 12×12 training grid and
persisted in `selection.json`. The universal evaluation combines prompt,
screen, and parser transfer in one run. `configs/desktop-multivlm.yaml` adds the
paper's joint three-VLM attack and evaluates it on all four Llama 3.2 Vision
models, including the held-out 90B pretrained model.

| Config | Paper coverage |
| --- | --- |
| `desktop-targeted.yaml`, `bluesky-targeted.yaml` | Table 1 |
| `desktop.yaml`, `bluesky.yaml` | Tables 2 and 8 |
| `desktop-multivlm.yaml` | Tables 4 and 9 |

All attack configs use the resource-exhaustion target in
`configs/targets/memory.txt` by default. To evaluate the website-navigation
target instead, pass `--target-file configs/targets/desktop-website.txt` with a
desktop config or `configs/targets/bluesky-website.txt` with a Bluesky config.
This repository evaluates generated first-turn actions without executing them.
The paper's multi-turn experiment in Table 3 is therefore not included.

The repository also includes:

- The exact 12/12 prompt and screen-split definitions used by the paper, plus
  the exact desktop screens tracked under `data/screens/desktop/`.
- Desktop and Bluesky parameters for both malicious targets.
- OmniParser and GroundingDINO + Tesseract evaluation paths.
- Bounded optimization, atomic current/best checkpoints, exact resume,
  append-only events, deterministic per-case sampling, and 8-bit RGB PNG
  artifacts.
- Long-form results plus paper-style ASR, sample standard deviation, and SEM.
- Regression tests for gradient averaging and sampled-generation metrics.

`adv_x_area` and `adv_y_area` define the allowed placement area. If that area
is larger than `adv_patch_size`, crafting uniformly resamples a valid integer
position for every setting, model, and image. Config validation emits an
explicit warning for this opt-in behavior, and evaluation places the saved PNG
at the center of the allowed area. The paper presets all use fixed placement
areas matching their patch sizes.

The public implementation incorporates a few refinements made during release review.
Newly crafted patches are therefore not expected to be bitwise identical to
historical paper runs. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Every configuration field and the patch-mode interactions are documented in
[CONFIGURATION.md](CONFIGURATION.md).

## WindowsAgentArena alignment

Evaluation follows the first planning turn of
[WindowsAgentArena's Navi agent](https://github.com/microsoft/WindowsAgentArena/tree/main/src/win-arena-container/client/mm_agents/navi):
screen parsing, set-of-marks annotation, WAA prompt construction, and
multimodal planning. Within a fixed configuration, MIP changes only the
screenshot before parsing; benign and patched observations use the same Llama
preprocessing, generation, and scoring path.

The paper defaults to the annotated screenshot alone to reduce differentiable
optimization cost. Its ablations found comparable attack effectiveness with
WAA's original two-image input containing both raw and annotated screenshots.

## Setup

The reference environment uses Python 3.11, PyTorch 2.5.1, CUDA 12.4, and
Transformers 4.45.2. Llama 3.2 Vision is publicly available under the Llama 
license, with access to the weights on HuggingFace being gated.

```bash
conda env create -f environment.yml
conda activate mip-against-agent
cp .env.example .env
# Edit .env: add a Hugging Face read token and choose a persistent HF_HOME.
python scripts/prepare_models.py --arial-font /path/to/arial.ttf
```

`mip-attack` and `scripts/prepare_models.py` automatically load `.env` without
overriding variables already exported by the shell. `.env.example` documents 
the token and cache variables. Models are normally stored under `$HF_HOME/hub`.

The 25 checksummed desktop screens are tracked directly in the repository:

```bash
python scripts/verify_assets.py
mip-attack validate --config configs/desktop.yaml
```

The social-media screenshots used in the paper are intentionally not
redistributed as they contain third-party public posts, account identifiers,
profile images, and post media. Redacting those pixels would also change the 
paper inputs. The Bluesky configs remain as a transparent record of the protocol,
but require independently obtained screenshots to be used at the configured paths. 
See [data/README.md](data/README.md).

The legacy OmniParser
detector checkpoint is AGPL-3.0 and uses a trusted PyTorch pickle format.
`prepare_models.py` verifies it before use. See
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream licenses.

## Run the paper protocols

Expose only the GPUs intended for the run. One 11B model fits on one modern
high-memory GPU; the 90B and multi-model settings require several GPUs.

Targeted attacks select and record one training prompt/screen pair:

```bash
mip-attack run \
  --config configs/desktop-targeted.yaml \
  --run-id desktop-targeted-memory
```

Universal attacks automatically evaluate all four prompt/screen split
combinations with both the seen OmniParser and held-out OSS parser:

```bash
mip-attack run \
  --config configs/desktop.yaml \
  --run-id desktop-universal-memory
```

Use the setting-specific website target for the second malicious behavior:

```bash
mip-attack run \
  --config configs/desktop.yaml \
  --target-file configs/targets/desktop-website.txt \
  --run-id desktop-universal-website
```

The VLM-universality experiment from Table 9 is a separate config:

```bash
mip-attack run \
  --config configs/desktop-multivlm.yaml \
  --run-id desktop-multivlm-memory
```

The same commands are resumable. Reuse the run ID and identical resolved
configuration after a preemption. `--seed` changes the targeted selection and
sampling streams.

With the default four temperatures, each targeted run writes 400 result rows,
each single-VLM universal run writes 18,432, and the four-model evaluation
writes 73,728. These counts include one greedy and five outputs at each of the
three stochastic temperatures.

Crafting and evaluation can also be scheduled separately:

```bash
mip-attack craft --config configs/desktop.yaml --run-id desktop-craft
mip-attack evaluate \
  --config configs/desktop.yaml \
  --patch-run runs/desktop-craft \
  --run-id desktop-evaluation
```

Regenerate aggregate files from a completed or partial evaluation with:

```bash
mip-attack summarize --evaluation-run runs/desktop-evaluation
```

## Results and run layout

`evaluation.csv` is the durable long-form record. `metrics.csv` and
`metrics.json` aggregate each model/parser/prompt-scope/screen-scope/temperature
cell. The `paper_metric` column reports mean ASR ± sample standard deviation;
SEM is a separate column.

```text
runs/<experiment-id>/
├── manifest.json
├── config.resolved.yaml
├── summary.json
├── attempts/<attempt-id>/{events.jsonl,summary.json}
├── craft/
│   ├── selection.json
│   └── examples/<patch-id>/
│       ├── artifacts/final_patch.png
│       ├── checkpoints/{latest.pt,best.pt}
│       └── attempts/<attempt-id>/diagnostics/
└── evaluation/
    ├── evaluation.csv
    ├── metrics.csv
    ├── metrics.json
    ├── prepared_screens/<scope>-<id>/...
    └── attempts/<attempt-id>/{events.jsonl,summary.json}
```

`latest.pt` contains the exact resumable optimizer state. `best.pt` separately
retains the best completed optimization boundary, ranked by full threshold
success, full argmax match, and then loss. `final_patch.png` is exported from
that best checkpoint, so a later optimizer excursion cannot discard a stronger
patch.

Reusing an identity with a different resolved config fails rather than mixing
results. Evaluation manifests also bind the patch, crafting config, and sampled
selection by SHA-256. Screen/parser preparation is cached, completed sample
identities are not regenerated, and stochastic generation seeds are derived
from each case's full identity. Slurm users can adapt
[scripts/slurm.sh](scripts/slurm.sh), which requests a preemption signal and
requeues checkpoint exit code 75.

## License

Project-authored code is licensed under the
[Apache License 2.0](LICENSE). Adapted and redistributed components retain
their upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). The desktop
screen images are not covered by the code license.

## Citation

Please consider citing the paper using [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{aichberger2025mip,
  title     = {MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents},
  author    = {Lukas Aichberger and Alasdair Paren and Guohao Li and Philip H. S. Torr and Yarin Gal and Adel Bibi},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025},
  doi       = {10.52202/085713-0625},
  url       = {https://papers.neurips.cc/paper_files/paper/2025/hash/1add4b4f8373c64504cfe4a3054700a1-Abstract-Conference.html}
}
```
