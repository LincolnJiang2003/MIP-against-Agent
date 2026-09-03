import unittest

import torch

from mip_attack.optimization import (
    backward_model_loss,
    best_patch_rank,
    capture_rng_state,
    exact_token_match,
    retain_best_patch,
    restore_rng_state,
    sampled_token_log_likelihoods,
    trim_generated_ids,
)


class BackwardMeanModelLossTests(unittest.TestCase):
    def test_accumulates_eight_settings_across_models(self):
        patch = torch.tensor(1.0, requires_grad=True)

        for setting_scale in range(1, 9):
            for model_loss in (
                setting_scale * patch,
                (setting_scale + 2) * patch,
            ):
                backward_model_loss(
                    model_loss,
                    model_count=2,
                    settings_per_update=8,
                )

        # One optimizer update uses the mean gradient across all eight settings.
        self.assertAlmostEqual(patch.grad.item(), 5.5)

    def test_rejects_empty_model_list(self):
        with self.assertRaisesRegex(ValueError, "model_count"):
            backward_model_loss(
                torch.tensor(1.0, requires_grad=True),
                model_count=0,
                settings_per_update=8,
            )

    def test_sample_metrics_follow_generated_tokens(self):
        step_scores = [
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[0.0, 2.0]]),
        ]
        sampled_ids = torch.tensor([1, 0])
        target_ids = torch.tensor([0, 1])

        sampled_log_likelihoods = sampled_token_log_likelihoods(
            step_scores, sampled_ids
        )
        expected = torch.log_softmax(torch.tensor([2.0, 0.0]), dim=0)[1]
        self.assertTrue(
            torch.allclose(sampled_log_likelihoods, expected.repeat(2))
        )
        self.assertFalse(exact_token_match(sampled_ids, target_ids))
        self.assertTrue(exact_token_match(torch.tensor([0, 1]), target_ids))
        self.assertFalse(exact_token_match(torch.tensor([9, 0, 1, 8]), target_ids))
        self.assertFalse(exact_token_match(torch.tensor([0, 1, 8]), target_ids))

    def test_sample_metrics_select_the_corresponding_generation_batch_row(self):
        step_scores = [
            torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
        ]
        sampled_ids = torch.tensor([1, 0])

        sampled_log_likelihoods = sampled_token_log_likelihoods(
            step_scores, sampled_ids, sequence_index=1
        )
        expected = torch.log_softmax(torch.tensor([0.0, 3.0]), dim=0)[1]

        self.assertTrue(
            torch.allclose(sampled_log_likelihoods, expected.repeat(2))
        )
        with self.assertRaisesRegex(ValueError, "generation batch"):
            sampled_token_log_likelihoods(
                step_scores, sampled_ids, sequence_index=2
            )

    def test_batched_generation_padding_is_trimmed_after_eos(self):
        output_ids = torch.tensor([7, 8, 2, 0, 0])

        self.assertTrue(
            torch.equal(
                trim_generated_ids(output_ids, eos_token_id=2),
                torch.tensor([7, 8, 2]),
            )
        )
        self.assertTrue(
            torch.equal(trim_generated_ids(output_ids, eos_token_id=None), output_ids)
        )

    def test_rng_state_can_be_restored_for_exact_replay(self):
        import random
        import numpy as np

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(3)

        random.random()
        np.random.random()
        torch.rand(3)
        restore_rng_state(state)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_best_patch_prefers_complete_success_then_argmax_then_loss(self):
        def metrics(loss, argmax=0, threshold=0, evaluations=8):
            return {
                "loss": loss,
                "argmax_matches": argmax,
                "targets_above_threshold": threshold,
                "evaluations": evaluations,
            }

        patch, saved_metrics, updated = retain_best_patch(
            torch.tensor([1.0]), metrics(1.0), None, None
        )
        self.assertTrue(updated)

        patch, saved_metrics, updated = retain_best_patch(
            torch.tensor([2.0]), metrics(0.5), patch, saved_metrics
        )
        self.assertTrue(updated)
        self.assertEqual(patch.item(), 2.0)

        patch, saved_metrics, updated = retain_best_patch(
            torch.tensor([3.0]), metrics(0.8, argmax=8), patch, saved_metrics
        )
        self.assertTrue(updated)
        self.assertEqual(patch.item(), 3.0)

        patch, saved_metrics, updated = retain_best_patch(
            torch.tensor([4.0]), metrics(0.9, argmax=8, threshold=8), patch, saved_metrics
        )
        self.assertTrue(updated)
        self.assertEqual(patch.item(), 4.0)

        _, _, updated = retain_best_patch(
            torch.tensor([5.0]), metrics(0.1, argmax=7, threshold=7), patch, saved_metrics
        )
        self.assertFalse(updated)
        self.assertGreater(best_patch_rank(saved_metrics), best_patch_rank(metrics(0.1)))

    def test_best_patch_snapshot_is_detached_from_candidate(self):
        candidate = torch.tensor([1.0], requires_grad=True)
        metrics = {
            "loss": 1.0,
            "argmax_matches": 0,
            "targets_above_threshold": 0,
            "evaluations": 1,
        }
        best, _, _ = retain_best_patch(candidate, metrics, None, None)
        with torch.no_grad():
            candidate.fill_(2.0)
        self.assertEqual(best.item(), 1.0)
        self.assertFalse(best.requires_grad)


if __name__ == "__main__":
    unittest.main()
