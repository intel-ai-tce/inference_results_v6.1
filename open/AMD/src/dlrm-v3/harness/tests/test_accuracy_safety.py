import unittest

from inference_harness.accuracy_safety import (
    accuracy_response_candidate_size,
    accuracy_response_copy_size,
    server_accuracy_duration_batches,
    should_run_zmq_warmups,
    should_run_optimized_compare,
    should_restore_inference_accuracy_targets,
    should_use_optimized_embed,
    should_use_uniform_targets_metadata,
    worker_warmup_steps_for_mode,
)


class AccuracySafetyTest(unittest.TestCase):
    def test_accuracy_mode_never_uses_optimized_embed(self):
        self.assertFalse(
            should_use_optimized_embed(
                optimized_lookup=True,
                optimized_disabled=False,
                is_inference=False,
                server_candidate_shape=True,
            )
        )

    def test_performance_uses_optimized_embed_only_for_server_shape(self):
        self.assertTrue(
            should_use_optimized_embed(
                optimized_lookup=True,
                optimized_disabled=False,
                is_inference=True,
                server_candidate_shape=True,
            )
        )
        self.assertFalse(
            should_use_optimized_embed(
                optimized_lookup=True,
                optimized_disabled=False,
                is_inference=True,
                server_candidate_shape=False,
            )
        )

    def test_compare_can_run_once_for_shape_validation(self):
        self.assertTrue(
            should_run_optimized_compare(
                optimized_compare=True,
                optimized_compares=0,
                optimized_compare_limit=1,
                is_inference=True,
                server_candidate_shape=True,
            )
        )
        self.assertFalse(
            should_run_optimized_compare(
                optimized_compare=True,
                optimized_compares=1,
                optimized_compare_limit=1,
                is_inference=True,
                server_candidate_shape=True,
            )
        )
        self.assertFalse(
            should_run_optimized_compare(
                optimized_compare=True,
                optimized_compares=0,
                optimized_compare_limit=1,
                is_inference=True,
                server_candidate_shape=False,
            )
        )
        self.assertFalse(
            should_run_optimized_compare(
                optimized_compare=True,
                optimized_compares=0,
                optimized_compare_limit=1,
                is_inference=False,
                server_candidate_shape=True,
            )
        )

    def test_uniform_targets_metadata_can_be_disabled_after_mismatch(self):
        self.assertTrue(
            should_use_uniform_targets_metadata(
                is_inference=True,
                uniform_targets_metadata=True,
                uniform_targets_metadata_disabled=False,
            )
        )
        self.assertFalse(
            should_use_uniform_targets_metadata(
                is_inference=True,
                uniform_targets_metadata=True,
                uniform_targets_metadata_disabled=True,
            )
        )
        self.assertFalse(
            should_use_uniform_targets_metadata(
                is_inference=False,
                uniform_targets_metadata=True,
                uniform_targets_metadata_disabled=False,
            )
        )

    def test_inference_accuracy_restores_reference_targets(self):
        self.assertTrue(
            should_restore_inference_accuracy_targets(
                perf_mode="accuracy",
                is_inference=True,
            )
        )
        self.assertFalse(
            should_restore_inference_accuracy_targets(
                perf_mode="performance",
                is_inference=True,
            )
        )
        self.assertFalse(
            should_restore_inference_accuracy_targets(
                perf_mode="accuracy",
                is_inference=False,
            )
        )

    def test_accuracy_mode_does_not_inherit_server_duration_warmup(self):
        self.assertEqual(server_accuracy_duration_batches("accuracy", 14297), 0)
        self.assertEqual(server_accuracy_duration_batches("performance", 14297), 14297)

    def test_accuracy_mode_skips_worker_predict_warmup(self):
        self.assertEqual(worker_warmup_steps_for_mode("accuracy", 60), 0)
        self.assertEqual(worker_warmup_steps_for_mode("performance", 60), 60)

    def test_accuracy_mode_skips_rocm_zmq_warmups(self):
        self.assertFalse(should_run_zmq_warmups(rocm_backend=True, mode="accuracy"))
        self.assertTrue(should_run_zmq_warmups(rocm_backend=True, mode="performance"))
        self.assertFalse(should_run_zmq_warmups(rocm_backend=False, mode="performance"))

    def test_accuracy_response_can_expand_for_test08(self):
        self.assertEqual(accuracy_response_candidate_size(32, 0), 32)
        self.assertEqual(accuracy_response_candidate_size(32, 2048), 2048)
        self.assertEqual(accuracy_response_candidate_size(2048, 32), 32)
        self.assertEqual(accuracy_response_copy_size(32, 2048), 32)
        self.assertEqual(accuracy_response_copy_size(2048, 32), 32)


if __name__ == "__main__":
    unittest.main()
