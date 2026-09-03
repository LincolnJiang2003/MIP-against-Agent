from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

from mip_attack.agent import NaviAgent, place_png_patch
from mip_attack.artifacts import (
    PREEMPT_EXIT_CODE,
    RunArtifacts,
    atomic_save_pil_png,
    atomic_write_json,
    preemption_requested,
    preemption_signal,
    relative_artifact_path,
    validate_id,
)
from mip_attack.config import centered_patch_ranges, screen_path, verify_assets
from mip_attack.protocol import EvaluationCondition, paper_conditions
from mip_attack.reporting import (
    RESULT_IDENTITY_COLUMNS,
    append_results,
    ensure_results_file,
    summarize_evaluation,
)
from mip_attack.runtime import (
    agent_kwargs,
    file_sha256,
    load_instructions,
    observation,
    open_rgb,
    prepare_run,
    release_cuda,
    seed_everything,
)


def _infer_patch_id(patch_run: Path, explicit: str | None) -> str:
    if explicit:
        return validate_id(explicit, "patch_id")
    candidates = sorted(
        path.parent.parent.name
        for path in (patch_run / "examples").glob("*/artifacts/final_patch.png")
    )
    if len(candidates) != 1:
        raise ValueError("--patch-id is required unless the patch run contains one patch")
    return candidates[0]


def _load_selection(patch_run: Path, patch_id: str) -> dict[str, Any]:
    path = patch_run / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"Crafting selection not found: {path}")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != 1 or selection.get("patch_id") != patch_id:
        raise ValueError(f"Invalid crafting selection: {path}")
    return selection


def _evaluation_inputs(
    config: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, list[tuple[str, str, str]]], dict[str, list[int]]]:
    attack = config["craft"]["attack"]
    if selection["attack"] != attack:
        raise ValueError("Evaluation attack does not match the crafted patch")

    training_prompts = load_instructions(config, "training")
    evaluation_prompts = load_instructions(config, "evaluation")
    if attack == "targeted":
        selected = selection["prompts"]
        selected_screens = selection["screens"]
        if len(selected) != 1 or len(selected_screens) != 1:
            raise ValueError("Targeted patches must record exactly one prompt and screen")
        selected_key = (selected[0]["domain"], selected[0]["example_id"])
        prompt_lookup = {
            (domain, example_id): instruction
            for domain, example_id, instruction in training_prompts
        }
        if selected_key not in prompt_lookup:
            raise ValueError("Selected targeted prompt is not in the training split")
        prompts = {
            "selected": [(*selected_key, prompt_lookup[selected_key])],
            "evaluation": evaluation_prompts,
        }
        screens = {
            "selected": [selected_screens[0]["screen_id"]],
            "evaluation": config["data"]["evaluation_screen_ids"],
        }
    else:
        prompts = {"training": training_prompts, "evaluation": evaluation_prompts}
        screens = {
            "training": config["data"]["training_screen_ids"],
            "evaluation": config["data"]["evaluation_screen_ids"],
        }
    return prompts, screens


def _prepared_paths(
    run_root: Path,
    screen_scope: str,
    screen_id: int,
    parser_origin: str | None = None,
) -> tuple[Path, Path | None, Path | None]:
    root = run_root / "prepared_screens" / f"{screen_scope}-{screen_id:02d}"
    patched_path = root / "patched.png"
    if parser_origin is None:
        return patched_path, None, None
    parser_root = root / parser_origin
    return patched_path, parser_root / "annotated.png", parser_root / "parsed.json"


def _unique_screens(attack: str, screens: dict[str, list[int]]) -> list[tuple[str, int]]:
    result = []
    seen = set()
    for condition in paper_conditions(attack):
        for screen_id in screens[condition.screen_scope]:
            identity = (condition.screen_scope, screen_id)
            if identity not in seen:
                result.append(identity)
                seen.add(identity)
    return result


def _prepare_screens(
    config: dict[str, Any],
    patch_path: Path,
    patch_craft: dict[str, Any],
    screen_entries: list[tuple[str, int]],
    parser_origins: list[str],
    artifacts: RunArtifacts,
    run_root: Path,
) -> None:
    evaluation_x_range, evaluation_y_range = centered_patch_ranges(
        patch_craft["adv_x_area"],
        patch_craft["adv_y_area"],
        patch_craft["adv_patch_size"],
    )
    for screen_scope, screen_id in screen_entries:
        patched_path, _, _ = _prepared_paths(run_root, screen_scope, screen_id)
        if not patched_path.is_file():
            screen = open_rgb(screen_path(config, screen_id))
            try:
                patched = place_png_patch(
                    screen,
                    patch_path,
                    evaluation_x_range,
                    evaluation_y_range,
                )
                atomic_save_pil_png(patched_path, patched)
                patched.close()
            finally:
                screen.close()

    for parser_origin in parser_origins:
        pending = [
            (scope, screen_id)
            for scope, screen_id in screen_entries
            if not all(
                path is not None and path.is_file()
                for path in _prepared_paths(run_root, scope, screen_id, parser_origin)
            )
        ]
        if not pending:
            continue

        kwargs = agent_kwargs(config)
        kwargs["som_origin"] = parser_origin
        parser_agent = NaviAgent(**kwargs, load_planner=False)
        try:
            for screen_scope, screen_id in pending:
                patched_path, annotated_path, parsed_path = _prepared_paths(
                    run_root, screen_scope, screen_id, parser_origin
                )
                patched = open_rgb(patched_path)
                try:
                    _, annotated, entities, rendering = parser_agent.prepare_screen(
                        patched, som_origin=parser_origin
                    )
                    atomic_save_pil_png(annotated_path, annotated)
                    atomic_write_json(
                        parsed_path, {"entities": entities, "rendering": rendering}
                    )
                    annotated.close()
                finally:
                    patched.close()
                artifacts.event(
                    "evaluation_screen_prepared",
                    parser=parser_origin,
                    screen_scope=screen_scope,
                    screen_id=screen_id,
                )
                if preemption_requested():
                    return
        finally:
            del parser_agent
            release_cuda()


def _load_prepared_screen(
    run_root: Path, screen_scope: str, screen_id: int, parser_origin: str
) -> tuple[Image.Image, Image.Image, Any, Any]:
    patched_path, annotated_path, parsed_path = _prepared_paths(
        run_root, screen_scope, screen_id, parser_origin
    )
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    return (
        open_rgb(patched_path),
        open_rgb(annotated_path),
        parsed["entities"],
        parsed["rendering"],
    )


def _case_seed(base_seed: int, identity: tuple[Any, ...]) -> int:
    payload = json.dumps([base_seed, *identity], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _pair_complete(
    completed: set[tuple[str, ...]],
    identity: tuple[Any, ...],
    temperatures: list[float],
    num_generations: int,
) -> bool:
    prefix = tuple(str(value) for value in identity)
    for temperature in temperatures:
        generation_count = 1 if temperature == 0 else num_generations
        for sample_index in range(generation_count):
            if prefix + (str(float(temperature)), str(sample_index)) not in completed:
                return False
    return True


def _model_complete(
    completed: set[tuple[str, ...]],
    config: dict[str, Any],
    patch_id: str,
    model: str,
    prompts: dict[str, list[tuple[str, str, str]]],
    screens: dict[str, list[int]],
    conditions: list[EvaluationCondition],
) -> bool:
    evaluation = config["evaluation"]
    for parser_origin in evaluation["parser_origins"]:
        for condition in conditions:
            for screen_id in screens[condition.screen_scope]:
                for domain, example_id, _ in prompts[condition.prompt_scope]:
                    identity = (
                        config["craft"]["attack"],
                        model,
                        parser_origin,
                        condition.prompt_scope,
                        condition.screen_scope,
                        domain,
                        example_id,
                        patch_id,
                        screen_id,
                    )
                    if not _pair_complete(
                        completed,
                        identity,
                        evaluation["temperatures"],
                        evaluation["num_sampled_generations"],
                    ):
                        return False
    return True


def _evaluation_generation_total(
    config: dict[str, Any],
    prompts: dict[str, list[tuple[str, str, str]]],
    screens: dict[str, list[int]],
    conditions: list[EvaluationCondition],
) -> int:
    evaluation = config["evaluation"]
    generations_per_case = sum(
        1 if temperature == 0 else evaluation["num_sampled_generations"]
        for temperature in evaluation["temperatures"]
    )
    cases_per_model_parser = sum(
        len(prompts[condition.prompt_scope]) * len(screens[condition.screen_scope])
        for condition in conditions
    )
    return (
        len(evaluation["models"])
        * len(evaluation["parser_origins"])
        * cases_per_model_parser
        * generations_per_case
    )


def _completed_exact_matches(results_path: Path) -> int:
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(
            row["target_match"].strip().lower() == "true"
            for row in csv.DictReader(handle)
        )


def _update_evaluation_progress(progress_bar, exact_matches: int, evaluated: int) -> None:
    asr = exact_matches / evaluated if evaluated else 0.0
    progress_bar.set_postfix(
        {"exact": f"{exact_matches}/{evaluated}", "ASR": f"{asr:.4f}"},
        refresh=False,
    )


def _finish_preempted(
    artifacts: RunArtifacts,
    results_path: Path,
    run_root: Path,
    completed: set[tuple[str, ...]],
    new_rows: int = 0,
) -> None:
    report = summarize_evaluation(results_path)
    artifacts.finish(
        "preempted",
        signal=preemption_signal(),
        evaluation_rows=len(completed),
        new_evaluation_rows=new_rows,
        metrics=relative_artifact_path(report["metrics_csv"], run_root),
    )


def _result_rows(
    identity: tuple[Any, ...], case_seed: int, results: dict, temperatures: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    event_samples = {}
    for temperature in temperatures:
        output = results[temperature]
        samples = []
        for sample_index, output_text in enumerate(output["output_str"]):
            target_match = output["sample_matches_target"][sample_index]
            rows.append(
                {
                    **dict(zip(RESULT_IDENTITY_COLUMNS[:9], identity)),
                    "temperature": float(temperature),
                    "sample_index": sample_index,
                    "case_seed": case_seed,
                    "target_match": target_match,
                    "geometric_mean_likelihood": output[
                        "sampled_geometric_mean_likelihoods"
                    ][sample_index],
                    "arithmetic_mean_likelihood": output[
                        "sampled_arithmetic_mean_likelihoods"
                    ][sample_index],
                    "output_text": output_text,
                }
            )
            samples.append(
                {
                    "sample_index": sample_index,
                    "output_ids": output["output_ids"][sample_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "target_match": target_match,
                }
            )
        event_samples[str(float(temperature))] = samples
    return rows, event_samples


def evaluate(
    config: dict[str, Any],
    patch_run: str | Path,
    run_id: str | None = None,
    patch_id: str | None = None,
) -> Path:
    asset_summary = verify_assets(config)
    seed_everything(config["seed"])

    patch_run = Path(patch_run).expanduser().resolve()
    patch_id = _infer_patch_id(patch_run, patch_id)
    patch_path = patch_run / "examples" / patch_id / "artifacts" / "final_patch.png"
    patch_config_path = patch_run / "config.resolved.yaml"
    if not patch_path.is_file():
        raise FileNotFoundError(f"Patch artifact not found: {patch_path}")
    patch_config = yaml.safe_load(patch_config_path.read_text(encoding="utf-8"))
    selection = _load_selection(patch_run, patch_id)
    prompts, screens = _evaluation_inputs(config, selection)
    conditions = paper_conditions(config["craft"]["attack"])

    artifacts, run_root = prepare_run(
        config,
        "evaluate",
        run_id,
        identity={
            "patch_sha256": file_sha256(patch_path),
            "patch_config_sha256": file_sha256(patch_config_path),
            "selection_sha256": file_sha256(patch_run / "selection.json"),
        },
    )
    evaluation = config["evaluation"]
    artifacts.event(
        "evaluation_started",
        patch_run=str(patch_run),
        patch_id=patch_id,
        attack=config["craft"]["attack"],
        parsers=evaluation["parser_origins"],
        models=evaluation["models"],
        conditions=[condition.__dict__ for condition in conditions],
        assets=asset_summary,
    )
    results_path = run_root / "evaluation.csv"
    completed = ensure_results_file(results_path)
    new_rows = 0

    _prepare_screens(
        config,
        patch_path,
        patch_config["craft"],
        _unique_screens(config["craft"]["attack"], screens),
        evaluation["parser_origins"],
        artifacts,
        run_root,
    )
    if preemption_requested():
        _finish_preempted(artifacts, results_path, run_root, completed)
        raise SystemExit(PREEMPT_EXIT_CODE)

    exact_matches = _completed_exact_matches(results_path)
    progress_bar = tqdm(
        desc="Evaluation",
        total=_evaluation_generation_total(config, prompts, screens, conditions),
        initial=len(completed),
        unit="gen",
        dynamic_ncols=True,
    )
    _update_evaluation_progress(progress_bar, exact_matches, len(completed))
    progress_bar.refresh()

    for model in evaluation["models"]:
        if _model_complete(completed, config, patch_id, model, prompts, screens, conditions):
            artifacts.event("evaluation_model_skipped", model=model, reason="completed")
            continue
        kwargs = agent_kwargs(config)
        kwargs["models"] = [model]
        agent = NaviAgent(**kwargs, load_parser=False)
        try:
            for parser_origin in evaluation["parser_origins"]:
                for condition in conditions:
                    for screen_id in screens[condition.screen_scope]:
                        prepared = _load_prepared_screen(
                            run_root, condition.screen_scope, screen_id, parser_origin
                        )
                        try:
                            for domain, example_id, instruction in prompts[
                                condition.prompt_scope
                            ]:
                                identity = (
                                    config["craft"]["attack"],
                                    model,
                                    parser_origin,
                                    condition.prompt_scope,
                                    condition.screen_scope,
                                    domain,
                                    example_id,
                                    patch_id,
                                    screen_id,
                                )
                                if _pair_complete(
                                    completed,
                                    identity,
                                    evaluation["temperatures"],
                                    evaluation["num_sampled_generations"],
                                ):
                                    continue
                                case_seed = _case_seed(config["seed"], identity)
                                seed_everything(case_seed)
                                results, _, _, _ = agent.predict_prepared(
                                    instruction=instruction,
                                    obs=observation(config, [prepared[0]], [instruction]),
                                    prepared_screen=prepared,
                                    malicious_generation=patch_config["target"]["generation"],
                                    temperatures=evaluation["temperatures"],
                                    num_generations=evaluation["num_sampled_generations"],
                                )
                                rows, samples = _result_rows(
                                    identity, case_seed, results, evaluation["temperatures"]
                                )
                                pending_rows = [
                                    row
                                    for row in rows
                                    if tuple(
                                        str(row[column])
                                        for column in RESULT_IDENTITY_COLUMNS
                                    )
                                    not in completed
                                ]
                                appended = append_results(results_path, rows, completed)
                                if appended != len(pending_rows):
                                    raise RuntimeError("Evaluation progress count mismatch")
                                new_rows += appended
                                exact_matches += sum(
                                    bool(row["target_match"]) for row in pending_rows
                                )
                                _update_evaluation_progress(
                                    progress_bar, exact_matches, len(completed)
                                )
                                progress_bar.update(appended)
                                artifacts.event(
                                    "evaluation_case_completed",
                                    model=model,
                                    parser=parser_origin,
                                    prompt_scope=condition.prompt_scope,
                                    screen_scope=condition.screen_scope,
                                    domain=domain,
                                    example_id=example_id,
                                    patch_id=patch_id,
                                    screen_id=screen_id,
                                    case_seed=case_seed,
                                    samples=samples,
                                )
                                if preemption_requested():
                                    progress_bar.close()
                                    _finish_preempted(
                                        artifacts,
                                        results_path,
                                        run_root,
                                        completed,
                                        new_rows,
                                    )
                                    raise SystemExit(PREEMPT_EXIT_CODE)
                        finally:
                            prepared[0].close()
                            prepared[1].close()
        finally:
            del agent
            release_cuda()

    progress_bar.close()
    report = summarize_evaluation(results_path)
    artifacts.finish(
        "completed",
        evaluation_rows=len(completed),
        new_evaluation_rows=new_rows,
        results=relative_artifact_path(results_path, run_root),
        metrics=relative_artifact_path(report["metrics_csv"], run_root),
        report=relative_artifact_path(report["metrics_json"], run_root),
    )
    return run_root
