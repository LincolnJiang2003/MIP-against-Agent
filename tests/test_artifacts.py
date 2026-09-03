import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from mip_attack.artifacts import (
    RunArtifacts,
    atomic_save_rgb_png,
    atomic_torch_save,
    install_exception_hook,
    relative_artifact_path,
    resolve_patch_path,
    torch_load,
)


class RunArtifactsTests(unittest.TestCase):
    def test_attempt_logs_are_append_only_and_summaries_are_atomic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run-1"
            run = RunArtifacts(root, "run-1", "attempt-1")
            run.initialize({"config_file": "config.yml", "seed": 42})
            run.event("optimization_boundary", step=8)
            run.finish("completed")

            events = [
                json.loads(line)
                for line in run.events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["run_attempt_started", "optimization_boundary", "run_attempt_finished"],
            )
            self.assertEqual(
                json.loads((root / "summary.json").read_text(encoding="utf-8"))["status"],
                "completed",
            )
            self.assertTrue(run.finished)

            resumed = RunArtifacts(root, "run-1", "attempt-2")
            resumed.initialize({"config_file": "config.yml", "seed": 42})
            self.assertTrue(run.events_path.exists())
            self.assertTrue(resumed.events_path.exists())

    def test_checkpoint_round_trip_and_patch_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run-1"
            run = RunArtifacts(root, "run-1", "attempt-1")
            crafting = run.crafting("example-1")
            self.assertEqual(
                crafting.best_checkpoint_path,
                root / "examples" / "example-1" / "checkpoints" / "best.pt",
            )
            checkpoint = root / "examples" / "example-1" / "checkpoints" / "latest.pt"
            atomic_torch_save(checkpoint, {"patch": torch.arange(4)})
            self.assertTrue(torch.equal(torch_load(checkpoint)["patch"], torch.arange(4)))

            legacy_patch = root / "example-1" / "example-1_final_patch.png"
            self.assertEqual(resolve_patch_path(root, "example-1"), legacy_patch)

            current_patch = root / "examples" / "example-1" / "artifacts" / "final_patch.png"
            current_patch.parent.mkdir(parents=True, exist_ok=True)
            current_patch.touch()
            self.assertEqual(resolve_patch_path(root, "example-1"), current_patch)

    def test_patch_is_saved_as_exact_eight_bit_rgb_png(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "patch.png"
            pixels = torch.tensor(
                [
                    [[0.0, 1.0], [128 / 255, 64 / 255]],
                    [[1.0, 0.0], [64 / 255, 128 / 255]],
                    [[0.5, 0.25], [1.0, 0.0]],
                ]
            )
            atomic_save_rgb_png(path, pixels)

            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (0, 255, 128))

    def test_relative_artifact_path_normalizes_parent_segments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lexical_root = root / "configs" / ".." / "runs" / "evaluation"
            metrics = root / "runs" / "evaluation" / "metrics.csv"
            self.assertEqual(
                relative_artifact_path(metrics, lexical_root),
                "metrics.csv",
            )

    def test_unhandled_exception_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run-1"
            run = RunArtifacts(root, "run-1", "attempt-1")
            run.initialize({"config_file": "config.yml", "seed": 42})

            original_hook = sys.excepthook
            delegated = []
            sys.excepthook = lambda *args: delegated.append(args)
            try:
                install_exception_hook(run)
                error = RuntimeError("boom")
                sys.excepthook(RuntimeError, error, None)
            finally:
                sys.excepthook = original_hook

            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["error_type"], "RuntimeError")
            self.assertEqual(summary["error_message"], "boom")
            self.assertEqual(len(delegated), 1)


if __name__ == "__main__":
    unittest.main()
