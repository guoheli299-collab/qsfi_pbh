from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import check_exchange_convergence as convergence
from qsfi_pbh.cumulants import StageEstimate


def _estimate(coefficient: float, power: int) -> StageEstimate:
    return StageEstimate(
        stage="exchange",
        raw_integral=-3.8e-4,
        raw_standard_error=2.0e-6,
        coefficient=coefficient,
        coefficient_standard_error=abs(coefficient) * 0.01,
        samples_per_replicate=2**power,
        completed_replicates=6,
        sobol_power_used=power,
        relative_standard_error=0.01,
        relative_error_target=0.05,
        precision_accepted=True,
        stop_reason="target_met",
        attempted_sobol_powers=(power,),
        sampling_history=(),
        excluded_ratio_samples=0,
        excluded_ratio_fraction=0.0,
        replicate_integrals=(-3.8e-4,) * 6,
    )


class ExchangeConvergenceCheckTests(unittest.TestCase):
    def test_case_matrix_covers_grid_and_sobol_refinements(self) -> None:
        self.assertEqual(
            [(case.sobol_power, case.ratio_nodes) for case in convergence.CASES],
            [(12, 128), (12, 256), (13, 256), (14, 256)],
        )

    def test_comparisons_apply_three_percent_limit(self) -> None:
        estimates = {
            "m12_n128": {"C4_exchange_hat": -100.0},
            "m12_n256": {"C4_exchange_hat": -101.0},
            "m13_n256": {"C4_exchange_hat": -102.0},
            "m14_n256": {"C4_exchange_hat": -102.5},
        }

        comparisons = convergence.build_comparisons(estimates, tolerance=0.03)

        self.assertEqual(
            list(comparisons),
            [
                "ratio_grid_128_to_256_at_m12",
                "sobol_m12_to_m13_at_n256",
                "sobol_m13_to_m14_at_n256",
                "overall_m12_n128_to_m14_n256",
            ],
        )
        self.assertTrue(all(item["passed"] for item in comparisons.values()))

        estimates["m14_n256"]["C4_exchange_hat"] = -110.0
        comparisons = convergence.build_comparisons(estimates, tolerance=0.03)
        self.assertFalse(comparisons["sobol_m13_to_m14_at_n256"]["passed"])
        self.assertFalse(comparisons["overall_m12_n128_to_m14_n256"]["passed"])

    @patch("check_exchange_convergence.estimate_stage")
    @patch("check_exchange_convergence.FullMomentumShapeEvaluator")
    @patch("check_exchange_convergence.build_ratio_function_bundle")
    @patch("check_exchange_convergence.full_momentum_settings_from_args")
    @patch("check_exchange_convergence.build_grid")
    @patch("check_exchange_convergence.make_args")
    def test_runner_uses_isolated_fixed_settings_and_shared_256_checkpoint(
        self,
        make_args: Mock,
        build_grid: Mock,
        settings_from_args: Mock,
        build_bundle: Mock,
        evaluator_type: Mock,
        estimate_stage: Mock,
    ) -> None:
        args = Mock()
        args.power_area = 0.01
        args.lambda2_over_H = 0.38
        args.force = True
        make_args.return_value = args
        grid = object()
        build_grid.return_value = grid
        base_settings = convergence.FullMomentumIntegrationSettings()
        settings_from_args.return_value = base_settings
        build_bundle.side_effect = ["bundle128", "bundle256"]
        evaluator_type.side_effect = ["evaluator128", "evaluator256"]
        estimate_stage.side_effect = [
            _estimate(-2.00, 12),
            _estimate(-2.02, 12),
            _estimate(-2.03, 13),
            _estimate(-2.04, 14),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "isolated"
            report = convergence.run_convergence(output_dir, tolerance=0.03)
            saved = json.loads((output_dir / convergence.REPORT_NAME).read_text())

        self.assertEqual(report, saved)
        self.assertTrue(report["passed"])
        self.assertEqual(report["sound_speed"]["sound_speed_ratio"], 3.0)
        self.assertEqual(build_bundle.call_count, 2)
        self.assertEqual(
            [call.kwargs["node_count"] for call in build_bundle.call_args_list],
            [128, 256],
        )
        self.assertEqual(estimate_stage.call_count, 4)
        settings = [call.args[2] for call in estimate_stage.call_args_list]
        self.assertEqual([item.sobol_power_exchange for item in settings], [12, 12, 13, 14])
        self.assertEqual([item.ratio_nodes for item in settings], [128, 256, 256, 256])
        self.assertTrue(all(item.sobol_min_replicates_four_point == 6 for item in settings))
        self.assertTrue(all(item.sobol_max_replicates_four_point == 6 for item in settings))
        self.assertTrue(all(item.sobol_max_power_increment_four_point == 0 for item in settings))
        checkpoints = [call.kwargs["checkpoint_path"] for call in estimate_stage.call_args_list]
        self.assertNotEqual(checkpoints[0], checkpoints[1])
        self.assertEqual(checkpoints[1], checkpoints[2])
        self.assertEqual(checkpoints[2], checkpoints[3])
        self.assertIn("isolated", str(args.cache_dir))
        self.assertFalse(args.force)

    def test_invalid_tolerance_is_rejected(self) -> None:
        for value in (0.0, -0.1, float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    convergence.build_comparisons({}, tolerance=value)


if __name__ == "__main__":
    unittest.main()
