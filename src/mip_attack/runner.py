from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image

from mip_attack.agent import NaviAgent
from mip_attack.artifacts import (
    PREEMPT_EXIT_CODE,
    atomic_write_json,
    preemption_requested,
    preemption_signal,
)
from mip_attack.config import resolve_path, screen_path, verify_assets
from mip_attack.evaluation import evaluate
from mip_attack.protocol import select_targeted_pair
from mip_attack.runtime import (
    agent_kwargs,
    load_instructions,
    observation,
    open_rgb,
    prepare_run,
    seed_everything,
)


def _attack_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    craft = dict(config["craft"])
    craft.pop("attack")
    initial_patch = craft["load_adversarial_patch"]
    if isinstance(initial_patch, str):
        craft["load_adversarial_patch"] = str(resolve_path(config, initial_patch))
    return craft


def _write_or_verify_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Persisted selection differs from current config: {path}")
    else:
        atomic_write_json(path, value)


def _crafting_inputs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[Image.Image]]:
    prompts = load_instructions(config, "training")
    screen_ids = config["data"]["training_screen_ids"]
    attack = config["craft"]["attack"]
    if attack == "targeted":
        prompt, screen_id = select_targeted_pair(prompts, screen_ids, config["seed"])
        prompts, screen_ids = [prompt], [screen_id]

    selection = {
        "schema_version": 1,
        "attack": attack,
        "patch_id": config["name"],
        "prompts": [
            {"split": "training", "domain": domain, "example_id": example_id}
            for domain, example_id, _ in prompts
        ],
        "screens": [
            {"split": "training", "screen_id": screen_id} for screen_id in screen_ids
        ],
        "parser_origins": config["craft"]["parser_origins"],
    }
    return (
        selection,
        [instruction for _, _, instruction in prompts],
        [open_rgb(screen_path(config, screen_id)) for screen_id in screen_ids],
    )


def craft(config: dict[str, Any], run_id: str | None = None) -> Path:
    asset_summary = verify_assets(config)
    seed_everything(config["seed"])
    artifacts, run_root = prepare_run(config, "craft", run_id)
    artifacts.event("assets_verified", **asset_summary)

    selection, instructions, screenshots = _crafting_inputs(config)
    _write_or_verify_json(run_root / "selection.json", selection)
    artifacts.event(
        "crafting_inputs_prepared",
        attack=selection["attack"],
        patch_id=selection["patch_id"],
        prompts=selection["prompts"],
        screens=selection["screens"],
        parser_origins=selection["parser_origins"],
        models=config["agent"]["models"],
    )
    try:
        agent = NaviAgent(**agent_kwargs(config))
        agent.attack(
            obs=observation(config, screenshots, instructions),
            example_id=selection["patch_id"],
            malicious_generation=config["target"]["generation"],
            run_artifacts=artifacts,
            **_attack_kwargs(config),
        )
    finally:
        for screenshot in screenshots:
            screenshot.close()

    if preemption_requested():
        artifacts.finish("preempted", signal=preemption_signal())
        raise SystemExit(PREEMPT_EXIT_CODE)
    artifacts.finish("completed", selection="selection.json")
    return run_root


def _completed_craft_run(run_root: Path, patch_id: str) -> bool:
    summary_path = run_root / "summary.json"
    patch_path = run_root / "examples" / patch_id / "artifacts" / "final_patch.png"
    selection_path = run_root / "selection.json"
    if not summary_path.is_file() or not patch_path.is_file() or not selection_path.is_file():
        return False
    return json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "completed"


def run_experiment(config: dict[str, Any], run_id: str | None = None) -> Path:
    """Craft and evaluate one complete paper protocol with resumable sub-runs."""
    artifacts, run_root = prepare_run(config, "run", run_id)
    child_config = copy.deepcopy(config)
    child_config["output_dir"] = str(run_root)
    craft_root = run_root / "craft"
    evaluation_root = run_root / "evaluation"
    patch_id = config["name"]

    try:
        if _completed_craft_run(craft_root, patch_id):
            artifacts.event("craft_skipped", reason="completed", path="craft")
        else:
            artifacts.event("craft_started", path="craft")
            craft(child_config, run_id="craft")
        artifacts.event("evaluation_started", path="evaluation")
        evaluate(
            child_config,
            patch_run=craft_root,
            run_id="evaluation",
            patch_id=patch_id,
        )
    except SystemExit as error:
        if error.code == PREEMPT_EXIT_CODE:
            artifacts.finish("preempted", signal=preemption_signal())
        else:
            artifacts.finish("failed", exit_code=error.code)
        raise
    except BaseException as error:
        artifacts.finish(
            "failed", error_type=type(error).__name__, error_message=str(error)
        )
        raise

    evaluation_summary = json.loads(
        (evaluation_root / "summary.json").read_text(encoding="utf-8")
    )
    artifacts.finish(
        "completed",
        craft="craft",
        evaluation="evaluation",
        evaluation_rows=evaluation_summary["evaluation_rows"],
        metrics="evaluation/metrics.csv",
        report="evaluation/metrics.json",
    )
    return run_root
