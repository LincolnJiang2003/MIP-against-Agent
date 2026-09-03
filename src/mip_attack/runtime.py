from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from mip_attack.artifacts import (
    RunArtifacts,
    atomic_write_text,
    install_exception_hook,
    install_preemption_handlers,
    new_attempt_id,
    validate_id,
)
from mip_attack.config import read_manifest, resolve_path, serializable_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run(
    config: dict[str, Any],
    command: str,
    run_id: str | None,
    identity: dict[str, Any] | None = None,
) -> tuple[RunArtifacts, Path]:
    if run_id is None:
        configured = os.environ.get("MIP_RUN_ID") or os.environ.get("SLURM_JOB_ID")
        run_id = configured or f"local-{new_attempt_id()}"
    run_id = validate_id(run_id, "run_id")
    run_root = resolve_path(config, config["output_dir"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    os.environ["MIP_RUN_DIR"] = str(run_root)

    snapshot = yaml.safe_dump(serializable_config(config), sort_keys=False)
    snapshot_path = run_root / "config.resolved.yaml"
    if snapshot_path.exists():
        if snapshot_path.read_text(encoding="utf-8") != snapshot:
            raise ValueError(f"Resolved config differs from existing run: {snapshot_path}")
    else:
        atomic_write_text(snapshot_path, snapshot)

    artifacts = RunArtifacts(run_root, run_id, new_attempt_id())
    artifacts.initialize(
        {
            "command": command,
            "config_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
            "config_snapshot": snapshot_path.name,
            "seed": config["seed"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            **(identity or {}),
        }
    )
    install_exception_hook(artifacts)
    install_preemption_handlers()
    return artifacts, run_root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_instructions(
    config: dict[str, Any], split: str
) -> list[tuple[str, str, str]]:
    data = config["data"]
    manifest = resolve_path(config, data[f"{split}_prompt_manifest"])
    examples_dir = resolve_path(config, data["examples_dir"])
    return [
        (
            domain,
            example_id,
            json.loads(
                (examples_dir / domain / f"{example_id}.json").read_text(encoding="utf-8")
            )["instruction"],
        )
        for domain, example_id in read_manifest(manifest)
    ]


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def observation(
    config: dict[str, Any], screenshots: list[Image.Image], instructions: list[str]
) -> dict[str, Any]:
    return {
        "screenshot": screenshots,
        "accessibility_tree": None,
        "terminal": None,
        "instruction": instructions,
        "window_title": config["data"]["window_title"],
        "window_rect": None,
        "window_image": None,
        "window_names_str": config["data"]["window_names"],
        "computer_clipboard": None,
        "human_input": None,
    }


def agent_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    agent = config["agent"]
    return {
        "models": agent["models"],
        "model_revisions": agent["model_revisions"],
        "som_origin": agent["som_origin"],
        "use_annotated_screen_only": agent["use_annotated_screen_only"],
        "prepare_text_images_individually": agent["prepare_text_images_individually"],
        "oss_box_threshold": agent["oss_box_threshold"],
        "parser_model_root": str(resolve_path(config, config["model_dir"])),
        "gpu_memory_reserve_gib": agent["gpu_memory_reserve_gib"],
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
