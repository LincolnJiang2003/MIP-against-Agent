import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from mip_attack.config import (
    PatchPositionAugmentationWarning,
    serializable_config,
    validate_config,
)
from mip_attack.evaluation import _update_evaluation_progress
from mip_attack.runner import evaluate, run_experiment


class FakeAgent:
    def __init__(self, **_kwargs):
        pass

    def prepare_screen(self, image, accessibility_tree=None, som_origin=None):
        return image, image.copy(), "entities", "rendering"

    def attack(self, obs, example_id, run_artifacts, **_kwargs):
        path = run_artifacts.crafting(example_id).final_patch_path
        Image.new("RGB", (1, 1), (1, 2, 3)).save(path)

    def predict_prepared(
        self,
        instruction,
        obs,
        prepared_screen,
        malicious_generation,
        temperatures,
        num_generations,
    ):
        return (
            {
                temperature: {
                    "output_str": [malicious_generation],
                    "output_ids": [torch.tensor([1])],
                    "sample_matches_target": [True],
                    "sampled_geometric_mean_likelihoods": [1.0],
                    "sampled_arithmetic_mean_likelihoods": [1.0],
                }
                for temperature in temperatures
            },
            None,
            None,
            None,
        )


def test_evaluation_progress_reports_exact_matches_and_four_decimal_asr():
    class Progress:
        def set_postfix(self, values, refresh):
            self.values = values
            self.refresh = refresh

    progress = Progress()
    _update_evaluation_progress(progress, exact_matches=3, evaluated=8)

    assert progress.values == {"exact": "3/8", "ASR": "0.3750"}
    assert progress.refresh is False


def _config(tmp_path: Path, attack: str) -> dict:
    examples = tmp_path / "examples"
    for split, domain, example_id in (
        ("training", "train-domain", "train-id"),
        ("evaluation", "eval-domain", "eval-id"),
    ):
        manifest = tmp_path / f"{split}.json"
        manifest.write_text(json.dumps({domain: [example_id]}), encoding="utf-8")
        example_path = examples / domain / f"{example_id}.json"
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text(json.dumps({"instruction": f"{split} prompt"}))
    for screen_id in (1, 2):
        Image.new("RGB", (2, 2), (screen_id, 0, 0)).save(tmp_path / f"screen-{screen_id}.png")

    config = {
        "schema_version": 1,
        "name": f"{attack}-patch",
        "seed": 42,
        "output_dir": str(tmp_path / "runs"),
        "model_dir": str(tmp_path / "models"),
        "data": {
            "examples_dir": str(examples),
            "training_prompt_manifest": str(tmp_path / "training.json"),
            "evaluation_prompt_manifest": str(tmp_path / "evaluation.json"),
            "screen_pattern": str(tmp_path / "screen-{screen_id}.png"),
            "training_screen_ids": [1],
            "evaluation_screen_ids": [2],
            "window_title": "Desktop",
            "window_names": "",
        },
        "agent": {
            "models": ["Llama-3.2-11B-Vision-Instruct"],
            "model_revisions": {
                "Llama-3.2-11B-Vision-Instruct": "0" * 40,
            },
            "som_origin": "omni",
            "oss_box_threshold": 0.22,
            "use_annotated_screen_only": True,
            "prepare_text_images_individually": False,
            "gpu_memory_reserve_gib": 2,
        },
        "target": {"source": "inline", "generation": "target"},
        "craft": {
            "attack": attack,
            "parser_origins": ["omni"],
            "load_adversarial_patch": False,
            "crop_patch_from_image": True,
            "max_num_patch_positions": 1,
            "resized_values": False,
            "adv_patch_size": [1, 1],
            "adv_x_area": [0, 1],
            "adv_y_area": [0, 1],
            "adv_optimizer": "adam",
            "adv_lr": 0.01,
            "betas": [0.9, 0.9],
            "increased_betas": [0.9, 0.9],
            "termination_likelihood": 0.99,
            "random_pert_intensity": 0.0,
            "settings_per_update": 1 if attack == "targeted" else 8,
            "max_optimizer_steps": 1,
            "checkpoint_interval": 1,
            "resume": True,
            "reset_betas_on_collapse": False,
            "epsilon": 1.0,
            "center_constraint_clipping_factor": 0.0,
            "keep_specific_color": False,
        },
        "evaluation": {
            "models": ["Llama-3.2-11B-Vision-Instruct"],
            "parser_origins": ["omni", "oss"] if attack == "universal" else ["omni"],
            "temperatures": [0.0],
            "num_sampled_generations": 5,
        },
        "_config_dir": str(tmp_path),
        "_config_path": str(tmp_path / "config.yaml"),
    }
    validate_config(config)
    return config


def _patch_run(tmp_path: Path, config: dict) -> Path:
    patch_run = tmp_path / "patch-run"
    patch_path = patch_run / "examples" / config["name"] / "artifacts" / "final_patch.png"
    patch_path.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), (1, 2, 3)).save(patch_path)
    (patch_run / "config.resolved.yaml").write_text(
        yaml.safe_dump(serializable_config(config)), encoding="utf-8"
    )
    selection = {
        "schema_version": 1,
        "attack": config["craft"]["attack"],
        "patch_id": config["name"],
        "prompts": [
            {"split": "training", "domain": "train-domain", "example_id": "train-id"}
        ],
        "screens": [{"split": "training", "screen_id": 1}],
        "parser_origins": ["omni"],
    }
    (patch_run / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return patch_run


def test_universal_evaluation_runs_all_split_and_parser_combinations(tmp_path, monkeypatch):
    config = _config(tmp_path, "universal")
    patch_run = _patch_run(tmp_path, config)
    monkeypatch.setattr("mip_attack.evaluation.NaviAgent", FakeAgent)

    run_root = evaluate(config, patch_run, run_id="evaluation")
    with (run_root / "evaluation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    combinations = {
        (row["prompt_scope"], row["screen_scope"], row["parser"]) for row in rows
    }

    assert len(rows) == 8
    assert len(combinations) == 8
    assert json.loads((run_root / "summary.json").read_text())["status"] == "completed"
    with (run_root / "metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 8

    resumed_root = evaluate(config, patch_run, run_id="evaluation")
    with (resumed_root / "evaluation.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 8

    patch_path = patch_run / "examples" / config["name"] / "artifacts" / "final_patch.png"
    Image.new("RGB", (1, 1), (9, 9, 9)).save(patch_path)
    with pytest.raises(ValueError, match="patch_sha256"):
        evaluate(config, patch_run, run_id="evaluation")


def test_targeted_evaluation_uses_selected_pair_and_paper_transfer_axes(tmp_path, monkeypatch):
    config = _config(tmp_path, "targeted")
    patch_run = _patch_run(tmp_path, config)
    monkeypatch.setattr("mip_attack.evaluation.NaviAgent", FakeAgent)

    run_root = evaluate(config, patch_run, run_id="evaluation")
    with (run_root / "evaluation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert {(row["prompt_scope"], row["screen_scope"]) for row in rows} == {
        ("selected", "selected"),
        ("evaluation", "selected"),
        ("selected", "evaluation"),
    }


def test_evaluation_centers_patch_inside_larger_crafting_area(tmp_path, monkeypatch):
    config = _config(tmp_path, "universal")
    for screen_id in (1, 2):
        Image.new("RGB", (3, 3), (screen_id, 0, 0)).save(
            tmp_path / f"screen-{screen_id}.png"
        )
    config["craft"]["adv_x_area"] = [0, 3]
    config["craft"]["adv_y_area"] = [0, 3]
    with pytest.warns(PatchPositionAugmentationWarning):
        validate_config(config)

    patch_run = _patch_run(tmp_path, config)
    monkeypatch.setattr("mip_attack.evaluation.NaviAgent", FakeAgent)
    run_root = evaluate(config, patch_run, run_id="evaluation")

    with Image.open(
        run_root / "prepared_screens" / "training-01" / "patched.png"
    ) as patched:
        assert patched.getpixel((1, 1)) == (1, 2, 3)
        assert patched.getpixel((0, 0)) == (1, 0, 0)


def test_complete_experiment_orchestrates_resumable_craft_and_evaluation(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, "targeted")
    monkeypatch.setattr("mip_attack.runner.NaviAgent", FakeAgent)
    monkeypatch.setattr("mip_attack.evaluation.NaviAgent", FakeAgent)

    run_root = run_experiment(config, run_id="paper-run")
    assert (run_root / "craft" / "selection.json").is_file()
    assert (run_root / "evaluation" / "metrics.csv").is_file()
    assert json.loads((run_root / "summary.json").read_text())["status"] == "completed"

    assert run_experiment(config, run_id="paper-run") == run_root
