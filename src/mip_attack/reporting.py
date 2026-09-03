from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from mip_attack.artifacts import atomic_write_json, atomic_write_text


RESULT_IDENTITY_COLUMNS = [
    "attack",
    "model",
    "parser",
    "prompt_scope",
    "screen_scope",
    "domain",
    "example_id",
    "patch_id",
    "screen_id",
    "temperature",
    "sample_index",
]
RESULT_COLUMNS = RESULT_IDENTITY_COLUMNS + [
    "case_seed",
    "target_match",
    "geometric_mean_likelihood",
    "arithmetic_mean_likelihood",
    "output_text",
]
METRIC_COLUMNS = [
    "attack",
    "patch_id",
    "model",
    "parser",
    "prompt_scope",
    "screen_scope",
    "temperature",
    "asr",
    "std",
    "sem",
    "paper_metric",
    "unit",
    "num_units",
    "num_prompts",
    "num_screens",
    "num_pairs",
    "num_samples",
    "num_successes",
]


def ensure_results_file(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(f"Unexpected evaluation CSV schema: {path}")
        return {
            tuple(row[column] for column in RESULT_IDENTITY_COLUMNS)
            for row in reader
        }


def append_results(
    path: Path,
    rows: Iterable[dict[str, Any]],
    completed: set[tuple[str, ...]],
) -> int:
    pending = []
    for row in rows:
        key = tuple(str(row[column]) for column in RESULT_IDENTITY_COLUMNS)
        if key not in completed:
            pending.append((row, key))
    if not pending:
        return 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writerows(row for row, _ in pending)
        handle.flush()
        os.fsync(handle.fileno())
    completed.update(key for _, key in pending)
    return len(pending)


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Invalid target_match value: {value!r}")
    return normalized == "true"


def _csv_text(columns: list[str], rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def summarize_evaluation(results_path: str | Path) -> dict[str, Any]:
    """Create paper-style ASR aggregates from the durable long-form results."""
    results_path = Path(results_path).expanduser().resolve()
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(f"Unexpected evaluation CSV schema: {results_path}")
        raw_rows = list(reader)

    condition_columns = RESULT_IDENTITY_COLUMNS[:5] + ["patch_id", "temperature"]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[tuple(row[column] for column in condition_columns)].append(row)

    metrics = []
    scope_order = {"selected": 0, "training": 0, "evaluation": 1}
    for key, rows in grouped.items():
        condition = dict(zip(condition_columns, key))
        pair_samples: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in rows:
            pair_samples[(row["domain"], row["example_id"], row["screen_id"])].append(
                float(_as_bool(row["target_match"]))
            )
        pair_rates = {pair: statistics.mean(values) for pair, values in pair_samples.items()}
        prompts = {(domain, example_id) for domain, example_id, _ in pair_rates}
        screens = {screen_id for _, _, screen_id in pair_rates}

        if len(prompts) > 1:
            unit = "prompt"
            unit_values: dict[tuple[str, str], list[float]] = defaultdict(list)
            for (domain, example_id, _), rate in pair_rates.items():
                unit_values[(domain, example_id)].append(rate)
        elif len(screens) > 1:
            unit = "screen"
            unit_values = defaultdict(list)
            for (_, _, screen_id), rate in pair_rates.items():
                unit_values[(screen_id, "")].append(rate)
        else:
            unit = "pair"
            unit_values = {("pair", ""): list(pair_rates.values())}

        rates = [statistics.mean(values) for values in unit_values.values()]
        asr = statistics.mean(rates)
        std = statistics.stdev(rates) if len(rates) > 1 else 0.0
        sem = std / math.sqrt(len(rates))
        metric = {
            **condition,
            "asr": asr,
            "std": std,
            "sem": sem,
            "paper_metric": f"{asr:.2f} ± {std:.2f}",
            "unit": unit,
            "num_units": len(rates),
            "num_prompts": len(prompts),
            "num_screens": len(screens),
            "num_pairs": len(pair_rates),
            "num_samples": len(rows),
            "num_successes": sum(_as_bool(row["target_match"]) for row in rows),
        }
        metrics.append(metric)

    metrics.sort(
        key=lambda row: (
            row["attack"],
            row["model"],
            row["parser"],
            scope_order[row["prompt_scope"]],
            scope_order[row["screen_scope"]],
            float(row["temperature"]),
        )
    )
    metrics_path = results_path.with_name("metrics.csv")
    report_path = results_path.with_name("metrics.json")
    atomic_write_text(metrics_path, _csv_text(METRIC_COLUMNS, metrics))
    report = {
        "schema_version": 1,
        "results": results_path.name,
        "aggregation": (
            "ASR is averaged per prompt and sample standard deviation is reported across "
            "prompts; when only screens vary, screens are the aggregation units."
        ),
        "num_result_rows": len(raw_rows),
        "metrics": metrics,
    }
    atomic_write_json(report_path, report)
    return {
        "num_result_rows": len(raw_rows),
        "num_metric_rows": len(metrics),
        "metrics_csv": metrics_path,
        "metrics_json": report_path,
    }
