# Reproducibility notes

## Paper protocol

The configs encode the paper's 12 training and 12 held-out prompt and screen
splits. The public repository includes the exact desktop screens; the
social-media screens are not redistributed for the rights and privacy reasons
described in `data/README.md`. Targeted runs uniformly sample one pair from the
12×12 training grid using the configured seed; `selection.json` makes the
sampled pair explicit and stable on resume. They evaluate the selected pair,
held-out prompts on its screen, and held-out screens with its prompt. Universal
runs optimize over every training prompt and screen, then evaluate all four
train/held-out prompt and screen combinations. The main universal configs do
this with both OmniParser and the held-out GroundingDINO + Tesseract parser in
one invocation.

Universal crafting uses OmniParser plus its mixed variant and eight randomly
shuffled settings per optimizer update. The default VLM is
`Llama-3.2-11B-Vision-Instruct`; its paper revision and the parser artifacts are
pinned. The exact Arial 2.82 rendering font is checksum-pinned but must be
supplied separately under its own license. Final patches are quantized and
stored as lossless 8-bit RGB PNGs.

`configs/desktop.yaml` uses the paper's crop-initialized absolute RGB
parameterization, fixed `adv_y_area: [530, 1530]`, epsilon 0.1, center factor
1.3, Adam betas `[0.9, 0.9]`, and the GroundingDINO evaluation threshold 0.12
recorded by the paper-submission config. The Bluesky parameters are recorded
in `configs/bluesky.yaml`. The targeted parameters are preserved in the two
`*-targeted.yaml` configs. `configs/desktop-multivlm.yaml` applies the same
crop-based parameterization while averaging the loss across three VLMs.

All shipped presets use the fixed patch coordinates recorded by the experiment
artifacts: desktop `x=[1120, 2120]`, `y=[530, 1530]`, and social media
`x=[1218, 2118]`, `y=[394, 1294]`. Position augmentation remains available for
custom experiments by making either placement area larger than the patch;
validation warns whenever it is enabled.

`configs/desktop-multivlm.yaml` reproduces the Table 9 model composition: the
11B instruction-tuned, 11B pretrained, and 90B instruction-tuned models are
joint crafting models; all three plus the held-out 90B pretrained model are
evaluated. The historical repository recorded only the exact 11B-Instruct
revision. The other three entries are pinned to their official Hugging Face
snapshots retrieved for this release. This prevents future drift but cannot, by
itself, prove byte-for-byte identity with the historical multi-VLM run.

## Metrics

`evaluation.csv` contains one row per sampled output and records strict equality
of the complete generated and target token sequences. `metrics.csv` and
`metrics.json` reproduce the
paper's aggregation: each prompt is first averaged across screens and
generations, then mean ASR and sample standard deviation are computed across
prompts. When only screens vary, screens are the aggregation units. The files
also report SEM explicitly so standard deviation and standard error cannot be
confused.

Greedy decoding produces one output per pair; stochastic temperatures produce
five in one generation batch per temperature. This is 16 outputs per pair
across the four configured temperatures, as used in the paper's
computational-expense accounting. Batching preserves the sampling distribution
and output count, but a fixed seed need not produce the same individual token
sequences as the historical sequence of five separate generation calls.

## Corrections in this release

The release audit found and corrected the following implementation issues:

1. The historical loop was configured for eight settings per update but did
   not backpropagate eight losses. This release accumulates and averages all
   eight settings and all configured models before each update.
2. At nonzero temperatures, evaluation records exact target matching and
   likelihood for the sampled tokens, not the per-step argmax sequence.
3. VLM parameters are frozen, models run in evaluation mode, and crafting uses
   `use_cache=False`. Input gradients are retained while model-gradient and KV
   cache memory are avoided.
4. Optimization is bounded, checkpointed atomically, and exactly resumable,
   including partial accumulation windows and random-number-generator state.
5. The effective historical beta behavior remains the default through
   `reset_betas_on_collapse: false`.
6. Final artifacts are explicitly rounded to 8-bit RGB PNG values before they
   are evaluated on the carrier image.
7. Evaluation sampling is seeded from each case's complete identity, so a
   resumed matrix produces the same missing samples regardless of completed
   cases.
8. Crafting retains a separate best-observed checkpoint and exports it as the
   final PNG; the exact current optimizer state remains independently
   resumable from `latest.pt`.

These corrections preserve the intended attack formulation but change the
optimization trajectory relative to the historical implementation. Historical
targeted runs also did not persist their sampled pair as a first-class
artifact. Use released paper artifacts, rather than recrafting, when exact
published outputs are required.

## Determinism boundary

The config, targeted selection, prompt split, public desktop screens, parser
weights, 11B-Instruct model revision, dependency versions, per-case evaluation
seeds, sampled token IDs, and optimizer state are recorded or pinned. Exact
floating-point replay can still depend on GPU model, CUDA/cuDNN kernels, model
sharding, and upstream gated-model availability.

The paper used a node with eight NVIDIA A100 GPUs. A universal single-model run
is expensive; the full multi-model experiment is substantially larger. Start
with `mip-attack craft --max-steps 1` to validate the installation without
launching the full evaluation matrix.
