from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import compute_results
from compute_results import production_stage_names
import make_figures
from make_figures import EXPECTED_FIGURE_PATHS
from qsfi_pbh import analysis, backend, propagator
from qsfi_pbh.config import ProductionConfig
from qsfi_pbh.cumulants import FullMomentumIntegrationSettings
from qsfi_pbh.figure_data import (
    EXPECTED_DATA_PATHS,
    FigureDataError,
    validate_production_metadata,
)


class OutputContractTest(unittest.TestCase):
    def test_production_config_coupling_axes_and_metadata(self) -> None:
        config = ProductionConfig()
        lambda3, lambda4 = config.coupling_axes()

        self.assertEqual(config.lambda2_over_h, 0.38)
        self.assertEqual(lambda3.size, 2001)
        self.assertEqual(lambda4.size, 2001)
        self.assertEqual(lambda3[0], -1.0)
        self.assertEqual(lambda3[-1], 1.0)
        self.assertEqual(lambda4[0], -1.0)
        self.assertEqual(lambda4[-1], 1.0)
        self.assertAlmostEqual(lambda3[1] - lambda3[0], 0.001, places=12)
        self.assertAlmostEqual(lambda4[1] - lambda4[0], 0.001, places=12)
        self.assertEqual((lambda4 == 0.0).sum(), 1)
        validate_production_metadata(config.metadata())

        mass_prefix = "nu" + "_scan_"
        peak_prefix = "log10" + "_peak_height_"
        for field in tuple(
            f"{prefix}{suffix}"
            for prefix in (mass_prefix, peak_prefix)
            for suffix in ("min", "max", "step")
        ):
            with self.subTest(field=field):
                self.assertFalse(hasattr(config, field))
        metadata = config.metadata()
        for suffix in ("range", "min", "max", "step"):
            self.assertNotIn(f"{peak_prefix}{suffix}", metadata)

    def test_lambda2_and_scan_defaults_are_consistent_across_entry_points(
        self,
    ) -> None:
        config = ProductionConfig()
        self.assertEqual(analysis.EXPECTED_LAMBDA2_OVER_H, config.lambda2_over_h)
        self.assertEqual(backend.LAMBDA2_OVER_H, config.lambda2_over_h)
        self.assertEqual(
            FullMomentumIntegrationSettings().lambda2_over_H,
            config.lambda2_over_h,
        )
        with patch("sys.argv", ["propagator.py"]):
            args = propagator.parse_args()
        self.assertEqual(args.lambda2_over_H, config.lambda2_over_h)
        self.assertEqual((args.lambda3_min, args.lambda3_max), (-1.0, 1.0))
        self.assertEqual((args.lambda4_min, args.lambda4_max), (-1.0, 1.0))
        self.assertEqual(args.nscan, 2001)

    def test_stale_or_altered_metadata_is_rejected(self) -> None:
        altered_values = {
            "production_contract_version": "qsfi-pbh-update13-production-v1",
            "lambda4_points": 201,
            "lambda4_range": [0.0, 0.2],
            "lambda4_step": 0.002,
            "lambda2_over_H": 0.8,
        }
        for field, value in altered_values.items():
            with self.subTest(field=field):
                metadata = ProductionConfig().metadata()
                metadata[field] = value
                with self.assertRaises(FigureDataError):
                    validate_production_metadata(metadata)

    def test_production_stage_names(self) -> None:
        self.assertEqual(
            production_stage_names(),
            ("r_0p3", "r_1", "r_3", "manuscript_outputs"),
        )

    def test_public_workflow_source_has_no_retired_scans(self) -> None:
        root = Path(__file__).resolve().parents[1]
        deleted_module = "nu" + "_scan.py"
        self.assertFalse((root / "qsfi_pbh" / deleted_module).exists())

        for relative_path in (
            Path("compute_results.py"),
            Path("make_figures.py"),
            Path("qsfi_pbh/config.py"),
            Path("qsfi_pbh/figure_data.py"),
            Path("README.md"),
        ):
            source = (root / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                self.assertNotIn("nu" + "_scan", source)
                self.assertNotIn("nu" + "_peak_height_scan", source)
                self.assertNotIn("r3_mixing_scan_contributions", source)
                self.assertNotIn("r3_relative_contribution_mixing", source)
        legacy_peak_axis = "log10" + "_Pzeta_peak_NG"
        figure_data_source = (root / "qsfi_pbh" / "figure_data.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(legacy_peak_axis, figure_data_source)
        self.assertFalse((root / "qsfi_pbh" / "mixing.py").exists())

    def test_manuscript_export_writes_four_datasets_and_reports_stacked_grids(
        self,
    ) -> None:
        nus = np.asarray((0.1, 0.3, 0.5, 1.0))
        lambda3 = np.asarray((-0.2, 0.2))
        lambda4 = np.asarray((-0.2, 0.2))
        cases = []
        for ratio, offset in ((0.3, 0.0), (1.0, 10.0), (3.0, 20.0)):
            values = np.arange(16, dtype=float).reshape(4, 2, 2) + offset
            cases.append(
                {
                    "sound_speed": SimpleNamespace(ratio=ratio),
                    "nus": nus,
                    "lambda3_over_H": lambda3,
                    "lambda4": lambda4,
                    "log10_beta_ratio": values,
                    "Pzeta_NG_over_Pzeta_G": values + 100.0,
                    "QNG": values,
                    "S3": values,
                    "S4": values,
                }
            )
        coefficient_arrays = {
            "sound_speed_ratio": np.asarray((0.3, 1.0, 3.0)),
            "nu_tilde": nus,
            **{
                name: np.ones((3, 4), dtype=float)
                for name in (
                    "C3_hat",
                    "C3_hat_standard_error",
                    "C4_contact_hat",
                    "C4_contact_hat_standard_error",
                    "C4_exchange_hat",
                    "C4_exchange_hat_standard_error",
                )
            },
        }
        expected_beta = np.asarray(
            [case["log10_beta_ratio"] for case in cases]
        )
        expected_peak = np.asarray(
            [case["Pzeta_NG_over_Pzeta_G"] for case in cases]
        )
        expected_s3 = np.asarray([case["S3"] for case in cases])
        expected_s4 = np.asarray([case["S4"] for case in cases])
        expected_qng = np.asarray([case["QNG"] for case in cases])
        config = ProductionConfig()
        output_root = Path("/tmp/manuscript-output")
        written_payloads: dict[Path, object] = {}

        def capture_figure_data(
            path: Path | str,
            *,
            arrays: object,
            **_: object,
        ) -> Path:
            destination = Path(path)
            written_payloads[destination] = arrays
            return destination

        with (
            patch.object(
                compute_results, "_coefficient_arrays", return_value=coefficient_arrays
            ),
            patch.object(
                compute_results,
                "write_figure_data",
                side_effect=capture_figure_data,
            ) as write_data,
            patch.object(
                compute_results,
                "export_r3_contribution_dataset",
                return_value=output_root
                / "manuscript"
                / "r3_cumulant_contributions.npz",
            ) as export_r3,
            patch.object(
                compute_results,
                "format_manuscript_numerical_results",
                return_value="report\n",
            ) as format_report,
            patch.object(
                compute_results, "write_and_print_report"
            ) as write_report,
        ):
            written = compute_results.export_manuscript_datasets(
                cases=cases,
                config=config,
                output_root=output_root,
            )

        self.assertEqual(
            written,
            (
                output_root / "manuscript" / "hatted_cumulant_coefficients.npz",
                output_root / "manuscript" / "r3_cumulant_contributions.npz",
                output_root
                / "manuscript"
                / "sound_speed_log10_beta_ratio_full.npz",
                output_root / "manuscript" / "sound_speed_comparison_full.npz",
            ),
        )
        self.assertEqual(write_data.call_count, 3)
        export_r3.assert_called_once_with(
            case=cases[2],
            config=config,
            output_root=output_root,
        )
        np.testing.assert_array_equal(
            format_report.call_args.kwargs["log10_beta_ratio"], expected_beta
        )
        np.testing.assert_array_equal(
            format_report.call_args.kwargs["peak_height_ratio"], expected_peak
        )
        formatted = format_report.call_args.kwargs
        np.testing.assert_array_equal(formatted["S3"], expected_s3)
        np.testing.assert_array_equal(formatted["S4"], expected_s4)
        np.testing.assert_array_equal(formatted["Q_NG"], expected_qng)
        self.assertIs(
            written_payloads[
                output_root
                / "manuscript"
                / "sound_speed_log10_beta_ratio_full.npz"
            ]["log10_beta_ratio"],
            formatted["log10_beta_ratio"],
        )
        self.assertIs(
            written_payloads[
                output_root
                / "manuscript"
                / "sound_speed_comparison_full.npz"
            ]["Pzeta_NG_over_Pzeta_G"],
            formatted["peak_height_ratio"],
        )
        self.assertIs(
            written_payloads[
                output_root
                / "manuscript"
                / "hatted_cumulant_coefficients.npz"
            ],
            coefficient_arrays,
        )
        self.assertIs(
            formatted["coefficients"],
            coefficient_arrays,
        )
        write_report.assert_called_once_with(
            output_root / compute_results.REPORT_RELATIVE_PATH,
            "report\n",
        )

    def test_manuscript_export_rejects_mismatched_case_axes_before_writing(
        self,
    ) -> None:
        nus = np.asarray((0.1, 0.3, 0.5, 1.0))
        lambda3 = np.asarray((-0.2, 0.2))
        lambda4 = np.asarray((-0.2, 0.2))
        values = np.zeros((4, 2, 2), dtype=float)
        cases = [
            {
                "sound_speed": SimpleNamespace(ratio=ratio),
                "nus": nus,
                "lambda3_over_H": lambda3,
                "lambda4": lambda4,
                "log10_beta_ratio": values,
                "Pzeta_NG_over_Pzeta_G": values,
                "S3": values,
                "S4": values,
                "QNG": values,
            }
            for ratio in (0.3, 1.0, 3.0)
        ]
        cases[1] = {
            **cases[1],
            "lambda3_over_H": np.asarray((-0.1, 0.2)),
        }

        with (
            patch.object(compute_results, "write_figure_data") as write_data,
            patch.object(
                compute_results, "export_r3_contribution_dataset"
            ) as export_r3,
            patch.object(
                compute_results, "write_and_print_report"
            ) as write_report,
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"case 1.*lambda3_over_H",
            ):
                compute_results.export_manuscript_datasets(
                    cases=cases,
                    config=ProductionConfig(),
                    output_root=Path("/tmp/manuscript-output"),
                )

        write_data.assert_not_called()
        export_r3.assert_not_called()
        write_report.assert_not_called()

    def test_manuscript_export_rejects_mismatched_cumulant_shape_before_writing(
        self,
    ) -> None:
        nus = np.asarray((0.1, 0.3, 0.5, 1.0))
        lambda3 = np.asarray((-0.2, 0.2))
        lambda4 = np.asarray((-0.2, 0.2))
        values = np.zeros((4, 2, 2), dtype=float)
        cases = [
            {
                "sound_speed": SimpleNamespace(ratio=ratio),
                "nus": nus,
                "lambda3_over_H": lambda3,
                "lambda4": lambda4,
                "log10_beta_ratio": values,
                "Pzeta_NG_over_Pzeta_G": values,
                "S3": values,
                "S4": values,
                "QNG": values,
            }
            for ratio in (0.3, 1.0, 3.0)
        ]
        cases[1] = {
            **cases[1],
            "S3": np.zeros((4, 2, 1), dtype=float),
        }

        with (
            patch.object(compute_results, "_coefficient_arrays") as coefficients,
            patch.object(compute_results, "write_figure_data") as write_data,
            patch.object(
                compute_results, "export_r3_contribution_dataset"
            ) as export_r3,
            patch.object(
                compute_results, "write_and_print_report"
            ) as write_report,
        ):
            with self.assertRaisesRegex(ValueError, r"case 1 S3.*shape"):
                compute_results.export_manuscript_datasets(
                    cases=cases,
                    config=ProductionConfig(),
                    output_root=Path("/tmp/manuscript-output"),
                )

        coefficients.assert_not_called()
        write_data.assert_not_called()
        export_r3.assert_not_called()
        write_report.assert_not_called()

    def test_expected_data_paths(self) -> None:
        self.assertEqual(
            EXPECTED_DATA_PATHS,
            (
                Path("manuscript/hatted_cumulant_coefficients.npz"),
                Path("manuscript/r3_cumulant_contributions.npz"),
                Path("manuscript/sound_speed_log10_beta_ratio_full.npz"),
                Path("manuscript/sound_speed_comparison_full.npz"),
            ),
        )

    def test_expected_figure_paths(self) -> None:
        self.assertEqual(
            EXPECTED_FIGURE_PATHS,
            (
                Path("r3_relative_contribution_bars.pdf"),
                Path("r3_relative_contribution_bars.svg"),
                Path("r3_relative_contribution_bars.png"),
                Path("sound_speed_comparison_full.pdf"),
                Path("sound_speed_comparison_full.svg"),
                Path("sound_speed_comparison_full.png"),
                Path("sound_speed_log10_beta_ratio_full.pdf"),
                Path("sound_speed_log10_beta_ratio_full.svg"),
                Path("sound_speed_log10_beta_ratio_full.png"),
            ),
        )

    def test_figure_renderer_uses_only_three_manuscript_datasets(self) -> None:
        ratios = np.asarray((0.3, 1.0, 3.0))
        nus = np.asarray((0.1, 0.3, 0.5, 1.0))
        lambda_axis = np.asarray((-0.2, 0.2))
        contribution_data = make_figures.FigureData(
            path=Path("r3_cumulant_contributions.npz"),
            metadata={"kind": "r3_cumulant_contributions"},
            arrays={
                "nu_tilde": nus,
                "lambda3_over_H": lambda_axis,
                "lambda4": lambda_axis,
                "contact_contribution": np.ones((4, 2, 2)),
                "exchange_contribution": np.ones((4, 2, 2)),
            },
        )
        beta_data = make_figures.FigureData(
            path=Path("sound_speed_log10_beta_ratio_full.npz"),
            metadata={"kind": "sound_speed_log10_beta_ratio_full"},
            arrays={
                "sound_speed_ratio": ratios,
                "nu_tilde": nus,
                "lambda3_over_H": lambda_axis,
                "lambda4": lambda_axis,
                "log10_beta_ratio": np.ones((3, 4, 2, 2)),
            },
        )
        peak_data = make_figures.FigureData(
            path=Path("sound_speed_comparison_full.npz"),
            metadata={"kind": "sound_speed_comparison_full"},
            arrays={
                "sound_speed_ratio": ratios,
                "nu_tilde": nus,
                "lambda3_over_H": lambda_axis,
                "lambda4": lambda_axis,
                "Pzeta_NG_over_Pzeta_G": np.ones((3, 4, 2, 2)),
            },
        )
        data_root = Path("/tmp/manuscript-data")
        output_root = Path("/tmp/manuscript-figures")

        def load_for_path(path: Path | str, *, expected_kind: str):
            mapping = {
                data_root / "manuscript" / "r3_cumulant_contributions.npz": contribution_data,
                data_root / "manuscript" / "sound_speed_log10_beta_ratio_full.npz": beta_data,
                data_root / "manuscript" / "sound_speed_comparison_full.npz": peak_data,
            }
            self.assertEqual(mapping[Path(path)].metadata["kind"], expected_kind)
            return mapping[Path(path)]

        def paths_for(stem: Path | str) -> tuple[Path, ...]:
            return tuple(Path(f"{stem}.{suffix}") for suffix in ("pdf", "svg", "png"))

        with (
            patch.object(make_figures, "load_figure_data", side_effect=load_for_path) as load_data,
            patch.object(
                make_figures.plotting,
                "draw_relative_contribution_bars",
                side_effect=lambda _arrays, stem: paths_for(stem),
            ) as bars,
            patch.object(
                make_figures.plotting,
                "draw_sound_speed_coupling_grid",
                side_effect=lambda _bundle, stem, **_kwargs: paths_for(stem),
            ) as grids,
        ):
            outputs = make_figures.generate_figures(
                data_root=data_root,
                output_root=output_root,
            )

        self.assertEqual(
            set(outputs),
            {output_root / path for path in EXPECTED_FIGURE_PATHS},
        )
        self.assertEqual(load_data.call_count, 3)
        bars.assert_called_once()
        self.assertEqual(grids.call_count, 2)
        self.assertEqual(
            [call.kwargs["expected_kind"] for call in load_data.call_args_list],
            [
                "r3_cumulant_contributions",
                "sound_speed_log10_beta_ratio_full",
                "sound_speed_comparison_full",
            ],
        )


if __name__ == "__main__":
    unittest.main()
