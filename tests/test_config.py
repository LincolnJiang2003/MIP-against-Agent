import json
from pathlib import Path

import pytest

from mip_attack.config import (
    centered_patch_ranges,
    load_config,
    read_target,
    read_manifest,
)


ROOT = Path(__file__).parents[1]


def test_target_file_terminator_is_not_part_of_exact_match(tmp_path):
    target = "action<|eot_id|>"
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    bare = tmp_path / "bare.txt"
    lf.write_bytes((target + "\n").encode())
    crlf.write_bytes((target + "\r\n").encode())
    bare.write_bytes(target.encode())

    assert read_target(lf) == target
    assert read_target(crlf) == target
    assert read_target(bare) == target


@pytest.mark.parametrize(
    "name",
    ["desktop", "bluesky", "desktop-targeted", "bluesky-targeted", "desktop-multivlm"],
)
def test_paper_configs_use_explicit_disjoint_splits(name):
    config = load_config(ROOT / f"configs/{name}.yaml")
    data = config["data"]
    assert data["training_screen_ids"] == list(range(1, 13))
    assert data["evaluation_screen_ids"] == list(range(13, 25))
    assert set(data["training_screen_ids"]).isdisjoint(data["evaluation_screen_ids"])
    train = read_manifest(ROOT / "data/splits/train.json")
    evaluation = read_manifest(ROOT / "data/splits/eval.json")
    assert len(train) == len(evaluation) == 12
    assert set(train).isdisjoint(evaluation)
    craft = config["craft"]
    assert craft["adv_x_area"][1] - craft["adv_x_area"][0] == craft[
        "adv_patch_size"
    ][1]
    assert craft["adv_y_area"][1] - craft["adv_y_area"][0] == craft[
        "adv_patch_size"
    ][0]


@pytest.mark.parametrize("name", ["desktop", "bluesky"])
def test_paper_configs_use_eight_settings_and_finite_resume(name):
    config = load_config(ROOT / f"configs/{name}.yaml")
    craft = config["craft"]
    assert craft["settings_per_update"] == 8
    assert craft["max_optimizer_steps"] == 100_000
    assert craft["checkpoint_interval"] == 10
    assert craft["resume"] is True
    assert craft["reset_betas_on_collapse"] is False
    assert craft["attack"] == "universal"
    assert config["evaluation"]["parser_origins"] == ["omni", "oss"]


@pytest.mark.parametrize("name", ["desktop-targeted", "bluesky-targeted"])
def test_targeted_configs_use_one_seeded_setting(name):
    config = load_config(ROOT / f"configs/{name}.yaml")
    assert config["craft"]["attack"] == "targeted"
    assert config["craft"]["settings_per_update"] == 1
    assert config["craft"]["parser_origins"] == ["omni"]
    assert config["evaluation"]["parser_origins"] == ["omni"]


def test_desktop_config_preserves_paper_crafting_parameters():
    config = load_config(ROOT / "configs/desktop.yaml")
    craft = config["craft"]
    assert craft["crop_patch_from_image"] is True
    assert craft["adv_y_area"] == [530, 1530]
    assert craft["betas"] == [0.9, 0.9]
    assert craft["increased_betas"] == [0.95, 0.95]
    assert craft["epsilon"] == 0.1
    assert craft["center_constraint_clipping_factor"] == 1.3
    assert config["target"]["source"] == "file:targets/memory.txt"
    assert centered_patch_ranges(
        craft["adv_x_area"], craft["adv_y_area"], craft["adv_patch_size"]
    ) == ([1120, 2120], [530, 1530])


def test_asset_manifest_covers_checked_in_desktop_screens():
    manifest = json.loads((ROOT / "data/assets-manifest.json").read_text())
    assert len(manifest["files"]) == 25
    assert sum(item["size"] for item in manifest["files"].values()) == 122_925_645
    assert all(
        path.startswith("data/screens/desktop/") for path in manifest["files"]
    )
    assert all((ROOT / path).is_file() for path in manifest["files"])


def test_multivlm_config_crafts_on_three_and_evaluates_four_models():
    config = load_config(ROOT / "configs/desktop-multivlm.yaml")
    single_model = load_config(ROOT / "configs/desktop.yaml")
    assert len(config["agent"]["models"]) == 3
    assert len(config["evaluation"]["models"]) == 4
    assert set(config["agent"]["model_revisions"]) == set(config["evaluation"]["models"])
    assert config["craft"] == single_model["craft"]
    assert config["evaluation"]["parser_origins"] == ["omni", "oss"]


@pytest.mark.parametrize("name", ["desktop", "bluesky", "desktop-multivlm"])
def test_universal_configs_use_paper_oss_threshold(name):
    config = load_config(ROOT / f"configs/{name}.yaml")
    assert config["agent"]["oss_box_threshold"] == 0.12
