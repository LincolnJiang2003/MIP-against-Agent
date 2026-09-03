from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class EvaluationCondition:
    prompt_scope: str
    screen_scope: str


def select_targeted_pair(
    prompts: Sequence[T], screen_ids: Sequence[int], seed: int
) -> tuple[T, int]:
    """Uniformly and reproducibly select one element of prompts × screens."""
    if not prompts or not screen_ids:
        raise ValueError("Targeted selection requires prompts and screens")
    pair_index = random.Random(seed).randrange(len(prompts) * len(screen_ids))
    prompt_index, screen_index = divmod(pair_index, len(screen_ids))
    return prompts[prompt_index], screen_ids[screen_index]


def paper_conditions(attack: str) -> list[EvaluationCondition]:
    if attack == "targeted":
        return [
            EvaluationCondition("selected", "selected"),
            EvaluationCondition("evaluation", "selected"),
            EvaluationCondition("selected", "evaluation"),
        ]
    if attack == "universal":
        return [
            EvaluationCondition(prompt_scope, screen_scope)
            for prompt_scope in ("training", "evaluation")
            for screen_scope in ("training", "evaluation")
        ]
    raise ValueError(f"Unsupported attack: {attack}")
