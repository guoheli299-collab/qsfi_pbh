from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import qsfi_pbh.reporting as reporting
from qsfi_pbh.reporting import (
    COEFFICIENT_FIELDS,
    REPORT_RELATIVE_PATH,
    format_manuscript_numerical_results,
    write_and_print_report,
)


class ReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ratios = np.asarray([0.3, 1.0, 3.0])
        self.nus = np.asarray([0.1, 0.3, 0.5, 1.0])
        self.lambda3 = np.asarray([-0.2, 0.0, 0.2])
        self.lambda4 = np.asarray([-0.2, 0.2])
        base = np.arange(6, dtype=float).reshape(2, 3)
        self.fixed_power = np.empty((3, 4, 2, 3), dtype=float)
        self.fixed_abundance = np.empty_like(self.fixed_power)
        for ratio_index in range(3):
            for nu_index in range(4):
                offset = 10.0 * ratio_index + nu_index
                self.fixed_power[ratio_index, nu_index] = base + offset
                self.fixed_abundance[ratio_index, nu_index] = 1.0 - 0.01 * (
                    base + offset
                )
        self.s3 = self.fixed_power + 100.0
        self.s4 = self.fixed_power + 200.0
        self.qng = self.fixed_power + 300.0
        shape = (3, 4)
        self.coefficients = {
            "C3_hat": np.full(shape, 1.0),
            "C3_hat_standard_error": np.full(shape, 0.01),
            "C4_contact_hat": np.full(shape, -2.0),
            "C4_contact_hat_standard_error": np.full(shape, 0.02),
            "C4_exchange_hat": np.full(shape, 3.0),
            "C4_exchange_hat_standard_error": np.full(shape, 0.03),
        }

    def _format(self) -> str:
        return format_manuscript_numerical_results(
            sound_speed_ratio=self.ratios,
            nu_tilde=self.nus,
            lambda3_over_H=self.lambda3,
            lambda4=self.lambda4,
            log10_beta_ratio=self.fixed_power,
            peak_height_ratio=self.fixed_abundance,
            S3=self.s3,
            S4=self.s4,
            Q_NG=self.qng,
            coefficients=self.coefficients,
        )

    def test_report_contains_two_extrema_tables_and_12_coefficient_rows(self) -> None:
        report = self._format()
        self.assertIn("[fixed_power_extrema]", report)
        self.assertIn("[fixed_abundance_extrema]", report)
        self.assertIn("[projected_coefficients]", report)
        self.assertEqual(report.count("\n0.3\t"), 12)
        self.assertEqual(report.count("\n1\t"), 12)
        self.assertEqual(report.count("\n3\t"), 12)
        coefficient_section = report.split("[projected_coefficients]\n", 1)[1]
        self.assertEqual(len(coefficient_section.strip().splitlines()) - 1, 12)
        self.assertIn("C3_hat_standard_error", report)
        self.assertIn("C4_contact_hat_standard_error", report)
        self.assertIn("C4_exchange_hat_standard_error", report)
        self.assertEqual(COEFFICIENT_FIELDS[-1], "C4_exchange_hat_standard_error")
        self.assertTrue(report.endswith("\n"))

    def test_extrema_coordinates_follow_panel_axes(self) -> None:
        report = self._format()
        fixed_power = report.split("[fixed_power_extrema]\n", 1)[1].split(
            "\n[fixed_abundance_extrema]", 1
        )[0]
        self.assertEqual(
            fixed_power.strip().splitlines()[0].split("\t"),
            [
                "r_s",
                "nu_tilde",
                "extremum",
                "lambda3_over_H",
                "lambda4",
                "S3",
                "S4",
                "Q_NG",
            ],
        )
        first_data_row = fixed_power.strip().splitlines()[1]
        self.assertEqual(
            first_data_row.split("\t")[3:5], ["+0.200", "+0.200"]
        )
        self.assertEqual(first_data_row.split("\t")[5:], ["105", "205", "305"])

        fixed_abundance = report.split("[fixed_abundance_extrema]\n", 1)[1].split(
            "\n[projected_coefficients]", 1
        )[0]
        self.assertEqual(
            fixed_abundance.strip().splitlines()[0].split("\t"),
            fixed_power.strip().splitlines()[0].split("\t"),
        )
        first_abundance_row = fixed_abundance.strip().splitlines()[1]
        self.assertEqual(
            first_abundance_row.split("\t")[3:5], ["+0.200", "+0.200"]
        )
        self.assertEqual(
            first_abundance_row.split("\t")[5:], ["105", "205", "305"]
        )

    def test_writer_saves_exactly_what_it_prints(self) -> None:
        report = self._format()
        stream = StringIO()
        with TemporaryDirectory() as directory:
            destination = Path(directory) / REPORT_RELATIVE_PATH
            returned = write_and_print_report(destination, report, stream=stream)
            self.assertEqual(returned, destination)
            self.assertEqual(destination.read_bytes(), report.encode("utf-8"))
            self.assertEqual(stream.getvalue(), report)

    def test_writer_cleans_temporary_file_when_write_fails(self) -> None:
        real_named_temporary_file = reporting.tempfile.NamedTemporaryFile

        class FailingContext:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._handle = real_named_temporary_file(*args, **kwargs)

            def __enter__(self) -> "FailingContext":
                self._handle.__enter__()
                return self

            def __exit__(self, *args: object) -> object:
                return self._handle.__exit__(*args)

            @property
            def name(self) -> str:
                return self._handle.name

            def write(self, report: str) -> int:
                del report
                raise OSError("synthetic write failure")

        def failing_named_temporary_file(*args: object, **kwargs: object) -> FailingContext:
            return FailingContext(*args, **kwargs)

        with TemporaryDirectory() as directory:
            destination = Path(directory) / REPORT_RELATIVE_PATH
            with patch.object(
                reporting.tempfile,
                "NamedTemporaryFile",
                failing_named_temporary_file,
            ):
                with self.assertRaisesRegex(OSError, "synthetic write failure"):
                    write_and_print_report(destination, "report")
            self.assertFalse(destination.exists())
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")), []
            )

    def test_all_nan_panel_is_rejected(self) -> None:
        self.fixed_power[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "no finite values"):
            self._format()

    def test_wrong_grid_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            format_manuscript_numerical_results(
                sound_speed_ratio=self.ratios,
                nu_tilde=self.nus,
                lambda3_over_H=self.lambda3,
                lambda4=self.lambda4,
                log10_beta_ratio=np.zeros((3, 4, 2, 2)),
                peak_height_ratio=self.fixed_abundance,
                S3=self.s3,
                S4=self.s4,
                Q_NG=self.qng,
                coefficients=self.coefficients,
            )

    def test_wrong_shaped_or_nonfinite_extremum_cumulant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "S3.*shape"):
            format_manuscript_numerical_results(
                sound_speed_ratio=self.ratios,
                nu_tilde=self.nus,
                lambda3_over_H=self.lambda3,
                lambda4=self.lambda4,
                log10_beta_ratio=self.fixed_power,
                peak_height_ratio=self.fixed_abundance,
                S3=np.zeros((3, 4, 2, 2)),
                S4=self.s4,
                Q_NG=self.qng,
                coefficients=self.coefficients,
            )

        nonfinite = self.qng.copy()
        nonfinite[0, 0, 1, 2] = np.inf
        with self.assertRaisesRegex(ValueError, "Q_NG.*finite"):
            format_manuscript_numerical_results(
                sound_speed_ratio=self.ratios,
                nu_tilde=self.nus,
                lambda3_over_H=self.lambda3,
                lambda4=self.lambda4,
                log10_beta_ratio=self.fixed_power,
                peak_height_ratio=self.fixed_abundance,
                S3=self.s3,
                S4=self.s4,
                Q_NG=nonfinite,
                coefficients=self.coefficients,
            )

    def test_missing_or_nonfinite_coefficient_is_rejected(self) -> None:
        missing = dict(self.coefficients)
        del missing["C4_exchange_hat"]
        with self.assertRaisesRegex(ValueError, "coefficient"):
            format_manuscript_numerical_results(
                sound_speed_ratio=self.ratios,
                nu_tilde=self.nus,
                lambda3_over_H=self.lambda3,
                lambda4=self.lambda4,
                log10_beta_ratio=self.fixed_power,
                peak_height_ratio=self.fixed_abundance,
                S3=self.s3,
                S4=self.s4,
                Q_NG=self.qng,
                coefficients=missing,
            )

        nonfinite = dict(self.coefficients)
        nonfinite["C3_hat"] = self.coefficients["C3_hat"].copy()
        nonfinite["C3_hat"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            format_manuscript_numerical_results(
                sound_speed_ratio=self.ratios,
                nu_tilde=self.nus,
                lambda3_over_H=self.lambda3,
                lambda4=self.lambda4,
                log10_beta_ratio=self.fixed_power,
                peak_height_ratio=self.fixed_abundance,
                S3=self.s3,
                S4=self.s4,
                Q_NG=self.qng,
                coefficients=nonfinite,
            )


if __name__ == "__main__":
    unittest.main()
