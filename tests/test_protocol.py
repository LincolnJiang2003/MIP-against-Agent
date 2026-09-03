import csv
import math

import pytest

from mip_attack.protocol import paper_conditions, select_targeted_pair
from mip_attack.reporting import RESULT_COLUMNS, summarize_evaluation


def test_targeted_pair_is_uniformly_seeded_and_reproducible():
    prompts = ["a", "b", "c"]
    screens = [1, 2, 3, 4]
    first = select_targeted_pair(prompts, screens, seed=42)
    assert first == select_targeted_pair(prompts, screens, seed=42)
    assert first[0] in prompts and first[1] in screens


def test_paper_conditions_cover_targeted_and_universal_protocols():
    assert [(row.prompt_scope, row.screen_scope) for row in paper_conditions("targeted")] == [
        ("selected", "selected"),
        ("evaluation", "selected"),
        ("selected", "evaluation"),
    ]
    assert {(row.prompt_scope, row.screen_scope) for row in paper_conditions("universal")} == {
        ("training", "training"),
        ("training", "evaluation"),
        ("evaluation", "training"),
        ("evaluation", "evaluation"),
    }


def test_summary_matches_paper_prompt_level_mean_and_standard_deviation(tmp_path):
    results_path = tmp_path / "evaluation.csv"
    rows = []
    for example_id, target_match in (("prompt-a", True), ("prompt-b", False)):
        for screen_id in (1, 2):
            rows.append(
                {
                    "attack": "universal",
                    "model": "model",
                    "parser": "omni",
                    "prompt_scope": "training",
                    "screen_scope": "training",
                    "domain": example_id,
                    "example_id": example_id,
                    "patch_id": "patch",
                    "screen_id": screen_id,
                    "temperature": 1.0,
                    "sample_index": 0,
                    "case_seed": 7,
                    "target_match": target_match,
                    "geometric_mean_likelihood": 0.5,
                    "arithmetic_mean_likelihood": 0.5,
                    "output_text": "output",
                }
            )
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = summarize_evaluation(results_path)
    with report["metrics_csv"].open("r", encoding="utf-8", newline="") as handle:
        metric = next(csv.DictReader(handle))

    assert float(metric["asr"]) == 0.5
    assert float(metric["std"]) == pytest.approx(math.sqrt(0.5))
    assert float(metric["sem"]) == pytest.approx(0.5)
    assert metric["paper_metric"] == "0.50 ± 0.71"
    assert metric["unit"] == "prompt"
    assert int(metric["num_units"]) == 2
