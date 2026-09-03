# Configuration reference

Paths in a configuration file are resolved relative to that file. Command-line
`--seed`, `--target-file`, and `--max-steps` values override the corresponding
configured values before a run is recorded.

## Patch parameterization

The optimized tensor is an absolute, 8-bit-quantized RGB replacement patch. Its
initialization `P0` depends on two fields:

| Configuration | `P0` | Composition |
| --- | --- | --- |
| `load_adversarial_patch: <path>` | Loaded RGB PNG | Replace the carrier region |
| `crop_patch_from_image: true` | Crop from the first model input image | Replace the carrier region |
| Both above are false | Solid white | Replace the carrier region |

Loading and cropping are mutually exclusive. After every update, each patch
channel is projected to

```text
[max(0, P0 - E), min(1, P0 + E)]
```

and rounded to an 8-bit value. `E` is the spatially modulated value of
`epsilon`. Patch values can move above or below their initial RGB values wherever
the `[0,1]` RGB boundary permits.

## Experiment and data

| Field | Meaning |
| --- | --- |
| `schema_version` | Configuration schema; currently `1`. |
| `name` | Patch identity used in selections and artifact paths. |
| `seed` | Seeds input selection, optimization order, placement, and evaluation sampling. |
| `output_dir` | Parent directory for run artifacts. |
| `model_dir` | Parser-model and font directory prepared by `scripts/prepare_models.py`. |
| `data.examples_dir` | Root containing WAA task JSON files. |
| `data.training_prompt_manifest` | Prompt IDs used for crafting and seen-prompt evaluation. |
| `data.evaluation_prompt_manifest` | Disjoint held-out prompt IDs. |
| `data.screen_pattern` | Python format string containing `{screen_id}` for screenshots. |
| `data.training_screen_ids` | Screens used for crafting and seen-screen evaluation. |
| `data.evaluation_screen_ids` | Disjoint held-out screens. |
| `data.window_title` | Window title inserted into the WAA planner message. |
| `data.window_names` | Window-list text inserted into the WAA planner message. |
| `target.file` | Exact malicious completion to optimize; one terminal file line ending is ignored. |

## Agent

| Field | Meaning |
| --- | --- |
| `agent.models` | VLMs whose mean loss is optimized jointly. |
| `agent.model_revisions` | Pinned Hugging Face commit for every crafting or evaluation VLM. |
| `agent.som_origin` | Primary screen parser; it must also occur in `craft.parser_origins`. |
| `agent.oss_box_threshold` | GroundingDINO confidence threshold for `oss` parser variants. |
| `agent.use_annotated_screen_only` | Use only the marked screen instead of WAA's raw-plus-marked pair. |
| `agent.prepare_text_images_individually` | Legacy alternate processor path; paper presets keep it `false`. |
| `agent.gpu_memory_reserve_gib` | Free memory withheld per visible GPU when constructing the model device map. |

## Crafting

| Field | Meaning |
| --- | --- |
| `craft.attack` | `targeted` selects one seeded prompt/screen pair; `universal` uses the training grid. |
| `craft.parser_origins` | Parser renderings included as crafting settings. |
| `craft.load_adversarial_patch` | `false` or an RGB image used as the initial absolute replacement patch. |
| `craft.crop_patch_from_image` | Initialize an absolute replacement patch from the first model input image. |
| `craft.max_num_patch_positions` | Number of patches placed per image; currently only `1` is supported. |
| `craft.resized_values` | Legacy coordinate-rescaling switch; currently unsupported and must be `false`. |
| `craft.adv_patch_size` | Patch `[height, width]` in original screenshot pixels. |
| `craft.adv_x_area` | Half-open horizontal placement area `[x_min, x_max]`. |
| `craft.adv_y_area` | Half-open vertical placement area `[y_min, y_max]`. |
| `craft.adv_optimizer` | `adam`, `sgd`, or unit-step gradient `sign`; `adv_lr` does not affect `sign`. |
| `craft.adv_lr` | Learning rate for Adam or SGD. |
| `craft.betas` | Initial Adam first- and second-moment coefficients. |
| `craft.increased_betas` | Adam coefficients selected after mean sequence likelihood exceeds `0.5`. |
| `craft.termination_likelihood` | Stop when every target-token likelihood exceeds this value for every accumulated setting and model. |
| `craft.random_pert_intensity` | Before optimization, subtract uniform noise up to this value from `P0`, then clamp to valid RGB. |
| `craft.settings_per_update` | Number of independently sampled settings averaged per update: `1` targeted or `8` universal. |
| `craft.max_optimizer_steps` | Hard upper bound on optimizer updates. |
| `craft.checkpoint_interval` | Number of optimizer updates between atomic current-state checkpoints. |
| `craft.resume` | Resume a compatible `latest.pt`, including optimizer, gradients, counters, and RNG state. |
| `craft.reset_betas_on_collapse` | Restore `betas` after a likelihood collapse; `false` preserves the historical effective behavior. |
| `craft.epsilon` | Per-channel L-infinity radius around `P0`, in normalized RGB units. |
| `craft.center_constraint_clipping_factor` | Applies `E = clip(epsilon * (1 - factor * normalized_radius), 0, 1)`; `0` is uniform. |
| `craft.keep_specific_color` | `false` or normalized `[R,G,B]`; matching initial pixels receive `E=0`. |

If either placement area is larger than the patch, crafting samples a valid
integer position independently for each setting, model, and input image.
Validation warns when this augmentation is enabled. Evaluation uses the centered
position.

## Evaluation

| Field | Meaning |
| --- | --- |
| `evaluation.models` | VLMs evaluated separately, including optional held-out models. |
| `evaluation.parser_origins` | Screen parsers evaluated separately. |
| `evaluation.temperatures` | Decoding temperatures; `0` is greedy, positive values sample. |
| `evaluation.num_sampled_generations` | Samples and inference batch size per positive temperature; greedy decoding always runs once. |

`target_match` is strict equality between the complete generated token sequence
and the target token sequence. Prefix, suffix, or containing matches do not
count.
