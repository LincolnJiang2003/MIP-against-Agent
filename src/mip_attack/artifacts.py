from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PREEMPT_EXIT_CODE = 75
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_preemption_signal: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_id(value: str, name: str) -> str:
    if not value or not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{name} must contain only letters, numbers, '.', '_', or '-'")
    return value


def resolve_run_id() -> str:
    explicit_id = os.environ.get("MIP_RUN_ID")
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if explicit_id:
        return validate_id(explicit_id, "MIP_RUN_ID")
    if slurm_id:
        return validate_id(slurm_id, "SLURM_JOB_ID")
    return f"local-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def new_attempt_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def active_run_root() -> Path:
    configured = os.environ.get("MIP_RUN_DIR")
    if configured:
        return Path(configured)
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        return Path("runs") / validate_id(slurm_id, "SLURM_JOB_ID")
    raise RuntimeError("No active run directory; set MIP_RUN_DIR")


def _atomic_replace(path: Path, write: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_replace(path, lambda handle: handle.write(value.encode("utf-8")))


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, serialized)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    _atomic_replace(path, lambda handle: torch.save(value, handle))


def atomic_save_rgb_png(path: Path | str, pixel_values: torch.Tensor) -> None:
    """Save a [3, H, W] tensor in [0, 1] as a lossless 8-bit RGB PNG."""
    if pixel_values.ndim != 3 or pixel_values.shape[0] != 3:
        raise ValueError("pixel_values must have shape [3, H, W]")
    if pixel_values.min() < 0 or pixel_values.max() > 1:
        raise ValueError("pixel_values must be in [0, 1]")

    image_uint8 = torch.round(pixel_values.detach().cpu() * 255).to(torch.uint8)
    image = Image.fromarray(image_uint8.permute(1, 2, 0).numpy())
    _atomic_replace(Path(path), lambda handle: image.save(handle, format="PNG"))


def atomic_save_pil_png(path: Path | str, image: Image.Image) -> None:
    """Atomically save a PIL image as a lossless 8-bit RGB PNG."""
    rgb_image = image.convert("RGB")
    _atomic_replace(Path(path), lambda handle: rgb_image.save(handle, format="PNG"))


def relative_artifact_path(path: Path | str, run_root: Path | str) -> str:
    """Return a stable run-relative path after normalizing both operands."""
    return str(Path(path).resolve().relative_to(Path(run_root).resolve()))


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        return torch.load(path, map_location="cpu")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def install_preemption_handlers() -> None:
    def request_preemption(signum: int, _frame: Any) -> None:
        global _preemption_signal
        _preemption_signal = signal.Signals(signum).name

    signal.signal(signal.SIGTERM, request_preemption)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_preemption)


def preemption_requested() -> bool:
    return _preemption_signal is not None


def preemption_signal() -> str | None:
    return _preemption_signal


class RunArtifacts:
    def __init__(self, root: Path, run_id: str, attempt_id: str):
        self.root = root
        self.run_id = validate_id(run_id, "run_id")
        self.attempt_id = validate_id(attempt_id, "attempt_id")
        self.attempt_dir = self.root / "attempts" / self.attempt_id
        self.events_path = self.attempt_dir / "events.jsonl"
        self.attempt_summary_path = self.attempt_dir / "summary.json"
        self.finished = False

        self.attempt_dir.mkdir(parents=True, exist_ok=False)

    def initialize(self, manifest: dict[str, Any]) -> None:
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing["run_id"] != self.run_id:
                raise ValueError(f"Run manifest identity mismatch in {manifest_path}")
            for key, value in manifest.items():
                if key == "slurm_job_id":
                    continue
                if existing.get(key) != value:
                    raise ValueError(f"Run manifest field '{key}' differs in {manifest_path}")
        else:
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "created_at": utc_now(),
                    **manifest,
                },
            )

        self.event("run_attempt_started", slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        self.write_attempt_summary("running")

    def event(self, event_type: str, **fields: Any) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now(),
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "event": event_type,
                **fields,
            },
        )

    def write_attempt_summary(self, status: str, **fields: Any) -> None:
        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "status": status,
            "updated_at": utc_now(),
            **fields,
        }
        atomic_write_json(self.attempt_summary_path, summary)
        atomic_write_json(self.root / "summary.json", summary)

    def finish(self, status: str, **fields: Any) -> None:
        example_summaries = {}
        for summary_path in sorted((self.root / "examples").glob("*/summary.json")):
            example_summaries[summary_path.parent.name] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        if example_summaries:
            fields["examples"] = example_summaries
        self.event("run_attempt_finished", status=status, **fields)
        self.write_attempt_summary(status, **fields)
        self.finished = True

    def crafting(self, example_id: str) -> "CraftingArtifacts":
        return CraftingArtifacts(self, validate_id(example_id, "example_id"))


class CraftingArtifacts:
    def __init__(self, run: RunArtifacts, example_id: str):
        self.run = run
        self.example_id = example_id
        self.example_dir = run.root / "examples" / example_id
        self.artifacts_dir = self.example_dir / "artifacts"
        self.attempt_dir = self.example_dir / "attempts" / run.attempt_id
        self.diagnostics_dir = self.attempt_dir / "diagnostics"
        self.checkpoint_path = self.example_dir / "checkpoints" / "latest.pt"
        self.best_checkpoint_path = self.example_dir / "checkpoints" / "best.pt"
        self.summary_path = self.example_dir / "summary.json"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    @property
    def final_patch_path(self) -> Path:
        return self.artifacts_dir / "final_patch.png"

    def event(self, event_type: str, **fields: Any) -> None:
        self.run.event(event_type, example_id=self.example_id, **fields)

    def write_summary(self, status: str, **fields: Any) -> None:
        atomic_write_json(
            self.summary_path,
            {
                "schema_version": 1,
                "run_id": self.run.run_id,
                "attempt_id": self.run.attempt_id,
                "example_id": self.example_id,
                "status": status,
                "updated_at": utc_now(),
                **fields,
            },
        )


def resolve_patch_path(run_root: Path, example_id: str) -> Path:
    current = run_root / "examples" / example_id / "artifacts" / "final_patch.png"
    if current.exists():
        return current
    return run_root / example_id / f"{example_id}_final_patch.png"


def install_exception_hook(run: RunArtifacts) -> None:
    """Persist an unhandled failure before delegating to Python's traceback hook."""
    previous_hook = sys.excepthook

    def record_failure(exc_type: type[BaseException], exc: BaseException, traceback: Any) -> None:
        if not run.finished:
            try:
                run.finish(
                    "failed",
                    error_type=exc_type.__name__,
                    error_message=str(exc),
                )
            except Exception:
                logging.exception("Could not persist failed run status")
        previous_hook(exc_type, exc, traceback)

    sys.excepthook = record_failure
