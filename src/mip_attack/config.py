from __future__ import annotations

import copy
import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "seed",
    "output_dir",
    "model_dir",
    "data",
    "agent",
    "target",
    "craft",
    "evaluation",
}

CRAFT_KEYS = {
    "attack",
    "parser_origins",
    "load_adversarial_patch",
    "crop_patch_from_image",
    "max_num_patch_positions",
    "resized_values",
    "adv_patch_size",
    "adv_x_area",
    "adv_y_area",
    "adv_optimizer",
    "adv_lr",
    "betas",
    "increased_betas",
    "termination_likelihood",
    "random_pert_intensity",
    "settings_per_update",
    "max_optimizer_steps",
    "checkpoint_interval",
    "resume",
    "reset_betas_on_collapse",
    "epsilon",
    "center_constraint_clipping_factor",
    "keep_specific_color",
}

SUPPORTED_MODELS = {
    "Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-11B-Vision",
    "Llama-3.2-90B-Vision-Instruct",
    "Llama-3.2-90B-Vision",
}
SUPPORTED_PARSERS = {"omni", "mixed-omni", "oss", "mixed-oss", "a11y"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class PatchPositionAugmentationWarning(UserWarning):
    """A crafting area larger than the patch enables random placement."""


def _require_keys(section: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{section}' must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ValueError(f"Missing '{section}' keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown '{section}' keys: {sorted(unknown)}")
    return value


def _pair(value: Any, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
        or value[0] >= value[1]
    ):
        raise ValueError(f"'{name}' must be an increasing pair of integers")
    return value


def centered_patch_ranges(
    x_area: list[int], y_area: list[int], patch_size: list[int]
) -> tuple[list[int], list[int]]:
    """Return the concrete centered patch box inside an allowed placement area."""
    patch_height, patch_width = patch_size
    x_slack = x_area[1] - x_area[0] - patch_width
    y_slack = y_area[1] - y_area[0] - patch_height
    if x_slack < 0 or y_slack < 0:
        raise ValueError("Patch exceeds the allowed placement area")
    x_start = x_area[0] + x_slack // 2
    y_start = y_area[0] + y_slack // 2
    return (
        [x_start, x_start + patch_width],
        [y_start, y_start + patch_height],
    )


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def read_target(path: Path) -> str:
    """Read a target while ignoring one conventional text-file line ending."""
    value = path.read_text(encoding="utf-8")
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def read_manifest(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt manifest must be a mapping: {path}")
    entries: list[tuple[str, str]] = []
    for domain, example_ids in data.items():
        if not isinstance(domain, str) or not isinstance(example_ids, list):
            raise ValueError(f"Invalid prompt manifest: {path}")
        entries.extend((domain, example_id) for example_id in example_ids)
    if not entries or any(not isinstance(item[1], str) for item in entries):
        raise ValueError(f"Prompt manifest is empty or invalid: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"Prompt manifest contains duplicates: {path}")
    return entries


def screen_path(config: dict[str, Any], screen_id: int) -> Path:
    pattern = config["data"]["screen_pattern"]
    return resolve_path(config, pattern.format(screen_id=screen_id))


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    missing = TOP_LEVEL_KEYS - set(config)
    unknown = set(config) - TOP_LEVEL_KEYS - {"_config_dir", "_config_path"}
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown)}")
    if config["schema_version"] != 1:
        raise ValueError("Only config schema_version 1 is supported")
    if not isinstance(config["name"], str) or not SAFE_NAME.fullmatch(config["name"]):
        raise ValueError("'name' must contain only letters, numbers, '.', '_', or '-'")
    if not isinstance(config["seed"], int):
        raise ValueError("'seed' must be an integer")

    data_keys = {
        "examples_dir",
        "training_prompt_manifest",
        "evaluation_prompt_manifest",
        "screen_pattern",
        "training_screen_ids",
        "evaluation_screen_ids",
        "window_title",
        "window_names",
    }
    data = _require_keys("data", config["data"], data_keys)
    for split in ("training", "evaluation"):
        ids = data[f"{split}_screen_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(item, int) for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"'{split}_screen_ids' must contain unique integers")
    if set(data["training_screen_ids"]) & set(data["evaluation_screen_ids"]):
        raise ValueError("Training and evaluation screens must be disjoint")

    agent_keys = {
        "models",
        "model_revisions",
        "som_origin",
        "oss_box_threshold",
        "use_annotated_screen_only",
        "prepare_text_images_individually",
        "gpu_memory_reserve_gib",
    }
    agent = _require_keys("agent", config["agent"], agent_keys)
    models = agent["models"]
    if not isinstance(models, list) or not models or not all(
        isinstance(model, str) and model for model in models
    ):
        raise ValueError("'agent.models' must be a non-empty list")
    if len(models) != len(set(models)) or not set(models).issubset(SUPPORTED_MODELS):
        raise ValueError("'agent.models' must contain unique supported Llama models")
    if agent["som_origin"] not in SUPPORTED_PARSERS:
        raise ValueError(f"Unsupported parser origin: {agent['som_origin']}")

    target = _require_keys("target", config["target"], {"source", "generation"})
    if not isinstance(target["source"], str) or not target["source"]:
        raise ValueError("'target.source' must be non-empty")
    if not isinstance(target["generation"], str) or not target["generation"]:
        raise ValueError("'target.generation' must be non-empty")

    craft = _require_keys("craft", config["craft"], CRAFT_KEYS)
    if craft["attack"] not in {"targeted", "universal"}:
        raise ValueError("'craft.attack' must be targeted or universal")
    for key in ("settings_per_update", "max_optimizer_steps", "checkpoint_interval"):
        if not isinstance(craft[key], int) or craft[key] <= 0:
            raise ValueError(f"'craft.{key}' must be a positive integer")
    if craft["max_num_patch_positions"] != 1:
        raise ValueError("Only one patch position is supported")
    patch_size = craft["adv_patch_size"]
    if (
        not isinstance(patch_size, list)
        or len(patch_size) != 2
        or not all(isinstance(item, int) and item > 0 for item in patch_size)
    ):
        raise ValueError("'craft.adv_patch_size' must be two positive integers")
    x_area = _pair(craft["adv_x_area"], "craft.adv_x_area")
    y_area = _pair(craft["adv_y_area"], "craft.adv_y_area")
    if x_area[1] - x_area[0] < patch_size[1]:
        raise ValueError("Patch width exceeds adv_x_area")
    if y_area[1] - y_area[0] < patch_size[0]:
        raise ValueError("Patch height exceeds adv_y_area")
    if (
        x_area[1] - x_area[0] != patch_size[1]
        or y_area[1] - y_area[0] != patch_size[0]
    ):
        area_size = [x_area[1] - x_area[0], y_area[1] - y_area[0]]
        patch_width_height = [patch_size[1], patch_size[0]]
        evaluation_x_range, evaluation_y_range = centered_patch_ranges(
            x_area, y_area, patch_size
        )
        warnings.warn(
            "Patch-position augmentation is enabled: craft.adv_x_area and "
            f"craft.adv_y_area define a {area_size[0]}x{area_size[1]} area, "
            f"larger than the {patch_width_height[0]}x{patch_width_height[1]} "
            "craft.adv_patch_size. "
            "Crafting will sample a fresh valid position for each setting, model, "
            "and image; evaluation will use the centered position "
            f"x={evaluation_x_range}, y={evaluation_y_range}. Confirm that the "
            "larger placement area is intentional.",
            PatchPositionAugmentationWarning,
            stacklevel=2,
        )
    parser_origins = craft["parser_origins"]
    if (
        not isinstance(parser_origins, list)
        or not parser_origins
        or len(parser_origins) != len(set(parser_origins))
        or not set(parser_origins).issubset(SUPPORTED_PARSERS)
    ):
        raise ValueError("'craft.parser_origins' must contain unique supported parsers")
    if agent["som_origin"] not in parser_origins:
        raise ValueError("'craft.parser_origins' must include agent.som_origin")
    expected_settings = 1 if craft["attack"] == "targeted" else 8
    if craft["settings_per_update"] != expected_settings:
        raise ValueError(
            f"'{craft['attack']}' attacks require settings_per_update={expected_settings}"
        )

    evaluation = _require_keys(
        "evaluation",
        config["evaluation"],
        {"models", "parser_origins", "temperatures", "num_sampled_generations"},
    )
    evaluation_models = evaluation["models"]
    if (
        not isinstance(evaluation_models, list)
        or not evaluation_models
        or len(evaluation_models) != len(set(evaluation_models))
        or not set(evaluation_models).issubset(SUPPORTED_MODELS)
    ):
        raise ValueError("'evaluation.models' must contain unique supported Llama models")
    evaluation_parsers = evaluation["parser_origins"]
    if (
        not isinstance(evaluation_parsers, list)
        or not evaluation_parsers
        or len(evaluation_parsers) != len(set(evaluation_parsers))
        or not set(evaluation_parsers).issubset(SUPPORTED_PARSERS)
    ):
        raise ValueError("'evaluation.parser_origins' must contain unique supported parsers")
    if (
        not isinstance(evaluation["temperatures"], list)
        or not evaluation["temperatures"]
        or not all(
            isinstance(value, (int, float)) and value >= 0
            for value in evaluation["temperatures"]
        )
        or len(evaluation["temperatures"])
        != len({float(value) for value in evaluation["temperatures"]})
    ):
        raise ValueError("evaluation.temperatures must be non-negative numbers")
    if not isinstance(evaluation["num_sampled_generations"], int) or evaluation[
        "num_sampled_generations"
    ] <= 0:
        raise ValueError("evaluation.num_sampled_generations must be positive")

    revisions = agent["model_revisions"]
    configured_models = set(models) | set(evaluation_models)
    if not isinstance(revisions, dict) or set(revisions) != configured_models:
        raise ValueError("Model revisions must be provided for every crafting/evaluation model")
    if any(
        not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
        for revision in revisions.values()
    ):
        raise ValueError("Model revisions must be 40-character commit hashes")


def load_config(
    path: str | Path, target_override: str | Path | None = None
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)

    target = config.get("target")
    if not isinstance(target, dict):
        raise ValueError("'target' must be a mapping")
    if target_override is not None:
        target_path = Path(target_override).expanduser().resolve()
        generation = read_target(target_path)
        source = f"override:{target_path.name}"
    elif "generation" in target:
        generation = target["generation"]
        source = target.get("source", "inline")
    elif set(target) == {"file"}:
        target_path = resolve_path(config, target["file"])
        generation = read_target(target_path)
        source = f"file:{target['file']}"
    else:
        raise ValueError("'target' must contain either 'file' or 'generation'")
    config["target"] = {"source": source, "generation": generation}
    validate_config(config)
    return config


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def iter_asset_paths(config: dict[str, Any]) -> Iterable[Path]:
    data = config["data"]
    yield resolve_path(config, data["training_prompt_manifest"])
    yield resolve_path(config, data["evaluation_prompt_manifest"])
    for split in ("training", "evaluation"):
        for screen_id in data[f"{split}_screen_ids"]:
            yield screen_path(config, screen_id)
    examples_dir = resolve_path(config, data["examples_dir"])
    for split in ("training", "evaluation"):
        manifest_path = resolve_path(config, data[f"{split}_prompt_manifest"])
        for domain, example_id in read_manifest(manifest_path):
            yield examples_dir / domain / f"{example_id}.json"


def verify_assets(config: dict[str, Any]) -> dict[str, Any]:
    missing = sorted({str(path) for path in iter_asset_paths(config) if not path.is_file()})
    if missing:
        raise FileNotFoundError("Missing experiment assets:\n" + "\n".join(missing))

    data = config["data"]
    train_manifest = resolve_path(config, data["training_prompt_manifest"])
    eval_manifest = resolve_path(config, data["evaluation_prompt_manifest"])
    train_examples = read_manifest(train_manifest)
    eval_examples = read_manifest(eval_manifest)
    if set(train_examples) & set(eval_examples):
        raise ValueError("Training and evaluation prompt manifests overlap")

    sizes = set()
    for split in ("training", "evaluation"):
        for screen_id in data[f"{split}_screen_ids"]:
            with Image.open(screen_path(config, screen_id)) as image:
                sizes.add(image.size)
                width, height = image.size
                if (
                    config["craft"]["adv_x_area"][1] > width
                    or config["craft"]["adv_y_area"][1] > height
                ):
                    raise ValueError("Crafting area exceeds screen dimensions")
    if len(sizes) != 1:
        raise ValueError(f"Screens must have identical dimensions, got {sorted(sizes)}")
    width, height = sizes.pop()

    return {
        "training_prompts": len(train_examples),
        "evaluation_prompts": len(eval_examples),
        "training_screens": len(data["training_screen_ids"]),
        "evaluation_screens": len(data["evaluation_screen_ids"]),
        "screen_size": [width, height],
    }
