from collections.abc import Mapping, Sequence
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def best_patch_rank(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    """Rank an observed patch by success, then argmax match, then training loss."""
    evaluations = int(metrics["evaluations"])
    if evaluations <= 0:
        raise ValueError("metrics.evaluations must be positive")
    loss = float(metrics["loss"])
    if not math.isfinite(loss):
        loss = math.inf
    targets_above_threshold = int(metrics["targets_above_threshold"])
    argmax_matches = int(metrics["argmax_matches"])
    if not 0 <= targets_above_threshold <= evaluations:
        raise ValueError("metrics.targets_above_threshold is out of range")
    if not 0 <= argmax_matches <= evaluations:
        raise ValueError("metrics.argmax_matches is out of range")
    return (
        float(targets_above_threshold == evaluations),
        float(argmax_matches == evaluations),
        -loss,
    )


def retain_best_patch(
    candidate_patch: torch.Tensor,
    candidate_metrics: Mapping[str, float],
    best_patch: torch.Tensor | None,
    best_metrics: Mapping[str, float] | None,
) -> tuple[torch.Tensor, dict[str, float], bool]:
    """Keep a detached snapshot of the best observed optimization boundary."""
    if best_metrics is None or best_patch_rank(candidate_metrics) > best_patch_rank(best_metrics):
        return candidate_patch.detach().clone(), dict(candidate_metrics), True
    if best_patch is None:
        raise ValueError("best_patch is required when best_metrics is set")
    return best_patch, dict(best_metrics), False


def backward_model_loss(
    model_loss: torch.Tensor,
    model_count: int,
    settings_per_update: int,
) -> torch.Tensor:
    """Backpropagate one VLM's contribution to the per-update mean loss.

    Calling this for every VLM and setting averages across both dimensions while
    allowing each VLM's activation graph to be released immediately.
    """
    if model_count <= 0:
        raise ValueError("model_count must be positive")
    if settings_per_update <= 0:
        raise ValueError("settings_per_update must be positive")

    (model_loss / (model_count * settings_per_update)).backward()
    return model_loss.detach()


def sampled_token_log_likelihoods(
    step_scores: Sequence[torch.Tensor],
    output_ids: torch.Tensor,
    sequence_index: int = 0,
) -> torch.Tensor:
    """Return log likelihoods for the tokens that generation actually sampled."""
    if not step_scores:
        raise ValueError("step_scores must contain at least one generation step")
    if sequence_index < 0:
        raise ValueError("sequence_index must be non-negative")

    output_ids = output_ids.flatten()
    length = min(len(step_scores), output_ids.numel())
    if length == 0:
        raise ValueError("output_ids must contain at least one generated token")

    if any(score.shape[0] <= sequence_index for score in step_scores[:length]):
        raise ValueError("sequence_index is outside the generation batch")
    scores = torch.stack(
        [score[sequence_index] for score in step_scores[:length]], dim=0
    )
    sampled_ids = output_ids[:length].to(scores.device)
    return F.log_softmax(scores, dim=-1).gather(1, sampled_ids[:, None]).squeeze(1)


def trim_generated_ids(
    output_ids: torch.Tensor,
    eos_token_id: int | Sequence[int] | None,
) -> torch.Tensor:
    """Remove batch padding after the first generated end-of-sequence token."""
    output_ids = output_ids.flatten()
    if eos_token_id is None:
        return output_ids
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not eos_ids:
        return output_ids
    eos_mask = torch.isin(output_ids, output_ids.new_tensor(eos_ids))
    eos_positions = torch.nonzero(eos_mask, as_tuple=False)
    if eos_positions.numel() == 0:
        return output_ids
    return output_ids[: int(eos_positions[0].item()) + 1]


def exact_token_match(output_ids: torch.Tensor, target_ids: torch.Tensor) -> bool:
    """Whether the complete generated token sequence exactly equals the target."""
    output_ids = output_ids.flatten()
    target_ids = target_ids.flatten().to(output_ids.device)
    if target_ids.numel() == 0:
        raise ValueError("target_ids must contain at least one token")
    return output_ids.numel() == target_ids.numel() and torch.equal(
        output_ids, target_ids
    )


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
