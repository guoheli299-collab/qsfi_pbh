from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from qsfi_pbh import plotting


class FigureNotationTest(unittest.TestCase):
    def test_relative_contribution_bars_omit_y_axis_title(self) -> None:
        arrays = {
            "nu_tilde": np.asarray([0.1, 0.3, 0.5, 1.0]),
            "S3": np.asarray([2.77, 2.27, 1.61, 0.077]),
            "S4_over_8": np.asarray([-1.40, -1.13, -0.81, 1.15]),
        }
        captured: dict[str, str] = {}

        def capture_ylabel(
            figure: object,
            output_path: Path | str,
        ) -> tuple[Path, ...]:
            del output_path
            captured["ylabel"] = figure.axes[0].get_ylabel()
            return ()

        with TemporaryDirectory() as directory:
            with patch.object(
                plotting,
                "save_publication_figure",
                side_effect=capture_ylabel,
            ):
                plotting.draw_relative_contribution_bars(
                    arrays,
                    Path(directory) / "relative_contribution_bars",
                )

        self.assertEqual(captured["ylabel"], "")

    def test_compact_sound_speed_grid_omits_trend_panel(self) -> None:
        panel = np.asarray([[-1.0, 0.5], [0.25, 1.0]], dtype=float)
        values = np.tile(panel, (3, 4, 1, 1))
        bundle = {
            "ratios": np.asarray([0.3, 1.0, 3.0]),
            "nus": np.asarray([0.1, 0.3, 0.5, 1.0]),
            "lambda3": np.asarray([-0.2, 0.2]),
            "lambda4": np.asarray([0.0, 0.2]),
            "log10_beta_ratio": values,
            "fixed_power_maxima": np.max(values, axis=(-2, -1)),
        }
        captured: dict[str, object] = {}

        def capture_layout(
            figure: object,
            output_path: Path | str,
        ) -> tuple[Path, ...]:
            del output_path
            captured["axis_count"] = len(figure.axes)
            captured["xlabels"] = [axis.get_xlabel() for axis in figure.axes]
            captured["ylabels"] = [axis.get_ylabel() for axis in figure.axes]
            return ()

        with TemporaryDirectory() as directory:
            with patch.object(
                plotting,
                "save_publication_figure",
                side_effect=capture_layout,
            ):
                plotting.draw_sound_speed_coupling_grid(
                    bundle,
                    Path(directory) / "compact_sound_speed_grid",
                    quantity="log10_beta_ratio",
                    include_trend=False,
                )

        self.assertEqual(captured["axis_count"], 15)
        self.assertNotIn(r"$r_s$", captured["xlabels"])
        row_labels = [
            label for label in captured["ylabels"] if r"$\lambda_4$" in label
        ]
        self.assertEqual(len(row_labels), 3)
        self.assertTrue(all(r"$r_s=" in label for label in row_labels))

    def test_sound_speed_grid_uses_one_independent_norm_per_row(self) -> None:
        rows = (
            np.asarray([[-1.0, 1.0], [-0.5, 0.5]]),
            np.asarray([[-10.0, 10.0], [-5.0, 5.0]]),
            np.asarray([[-100.0, 100.0], [-50.0, 50.0]]),
        )
        values = np.asarray(
            [np.tile(row, (4, 1, 1)) for row in rows],
            dtype=float,
        )
        bundle = {
            "ratios": np.asarray([0.3, 1.0, 3.0]),
            "nus": np.asarray([0.1, 0.3, 0.5, 1.0]),
            "lambda3": np.asarray([-0.2, 0.2]),
            "lambda4": np.asarray([-0.2, 0.2]),
            "log10_beta_ratio": values,
            "fixed_power_maxima": np.max(values, axis=(-2, -1)),
        }
        captured: dict[str, object] = {}

        def capture_norms(
            figure: object,
            output_path: Path | str,
        ) -> tuple[Path, ...]:
            del output_path
            heatmap_axes = figure.axes[:12]
            norms = [axis.collections[0].norm for axis in heatmap_axes]
            captured["axis_count"] = len(figure.axes)
            captured["norms"] = norms
            return ()

        with TemporaryDirectory() as directory:
            with patch.object(
                plotting,
                "save_publication_figure",
                side_effect=capture_norms,
            ):
                plotting.draw_sound_speed_coupling_grid(
                    bundle,
                    Path(directory) / "row_scaled_grid",
                    quantity="log10_beta_ratio",
                    include_trend=False,
                )

        norms = captured["norms"]
        self.assertEqual(captured["axis_count"], 15)
        for row_index in range(3):
            row_norms = norms[4 * row_index : 4 * (row_index + 1)]
            self.assertTrue(all(norm is row_norms[0] for norm in row_norms))
        self.assertIsNot(norms[0], norms[4])
        self.assertIsNot(norms[4], norms[8])
        self.assertEqual(
            [
                (norm.vmin, norm.vcenter, norm.vmax)
                for norm in norms[::4]
            ],
            [
                (-1.0, 0.0, 1.0),
                (-10.0, 0.0, 10.0),
                (-100.0, 0.0, 100.0),
            ],
        )

    def test_sound_speed_grid_hatches_nonfinite_qng_exclusions(self) -> None:
        panel = np.asarray([[-1.0, 1.0], [-0.5, 0.5]], dtype=float)
        values = np.tile(panel, (3, 4, 1, 1))
        values[2, :, 0, 0] = np.nan
        bundle = {
            "ratios": np.asarray([0.3, 1.0, 3.0]),
            "nus": np.asarray([0.1, 0.3, 0.5, 1.0]),
            "lambda3": np.asarray([-0.2, 0.2]),
            "lambda4": np.asarray([-0.2, 0.2]),
            "log10_beta_ratio": values,
            "fixed_power_maxima": np.nanmax(values, axis=(-2, -1)),
        }
        captured: dict[str, object] = {}

        def capture_hatches(
            figure: object,
            output_path: Path | str,
        ) -> tuple[Path, ...]:
            del output_path
            heatmap_axes = figure.axes[:12]
            captured["hatches"] = [
                [
                    hatch
                    for collection in axis.collections[1:]
                    for hatch in getattr(collection, "hatches", ())
                    if hatch
                ]
                for axis in heatmap_axes
            ]
            captured["figure_legend_count"] = len(figure.legends)
            return ()

        with TemporaryDirectory() as directory:
            with patch.object(
                plotting,
                "save_publication_figure",
                side_effect=capture_hatches,
            ):
                plotting.draw_sound_speed_coupling_grid(
                    bundle,
                    Path(directory) / "hatched_grid",
                    quantity="log10_beta_ratio",
                    include_trend=False,
                )

        hatches = captured["hatches"]
        self.assertTrue(
            all(not panel_hatches for panel_hatches in hatches[:8])
        )
        self.assertTrue(
            all("////" in panel_hatches for panel_hatches in hatches[8:])
        )
        self.assertEqual(captured["figure_legend_count"], 0)

    def test_reference_aware_ticks_label_both_sides_of_peak_ratio(self) -> None:
        lower = 2.9e-5
        upper = 7.4
        ticks = plotting._reference_aware_colorbar_ticks(
            np.asarray([0.0, 2.0, 4.0, 6.0, 8.0]),
            lower=lower,
            reference=1.0,
            upper=upper,
        )

        np.testing.assert_allclose(
            ticks,
            np.asarray(
                [
                    lower,
                    0.5 * (lower + 1.0),
                    1.0,
                    2.0,
                    4.0,
                    6.0,
                ]
            ),
        )

    def test_colorbar_tick_labels_preserve_integer_trailing_zeroes(self) -> None:
        labels = plotting._format_colorbar_tick_labels(
            np.asarray([-400.0, -200.0, 0.0, 35.0, 70.0])
        )

        self.assertEqual(labels, ["-400", "-200", "0", "35", "70"])

    def test_peak_ratio_colorbar_applies_reference_aware_ticks(self) -> None:
        lower = 2.9e-5
        upper = 7.4
        rows = (
            np.asarray([[0.996, 1.004], [0.998, 1.002]]),
            np.asarray([[0.95, 1.05], [0.975, 1.025]]),
            np.asarray([[lower, upper], [0.5, 1.0]]),
        )
        values = np.asarray(
            [np.tile(row, (4, 1, 1)) for row in rows],
            dtype=float,
        )
        bundle = {
            "ratios": np.asarray([0.3, 1.0, 3.0]),
            "nus": np.asarray([0.1, 0.3, 0.5, 1.0]),
            "lambda3": np.asarray([-0.2, 0.2]),
            "lambda4": np.asarray([-0.2, 0.2]),
            "Pzeta_NG_over_Pzeta_G": values,
            "fixed_abundance_minima": np.min(values, axis=(-2, -1)),
        }
        captured: dict[str, object] = {}

        def capture_ticks(
            figure: object,
            output_path: Path | str,
        ) -> tuple[Path, ...]:
            del output_path
            figure.canvas.draw()
            captured["bottom_colorbar_ticks"] = np.asarray(
                figure.axes[14].get_yticks(),
                dtype=float,
            )
            captured["top_colorbar_labels"] = [
                label.get_text()
                for label in figure.axes[12].get_yticklabels()
            ]
            captured["bottom_colorbar_labels"] = [
                label.get_text()
                for label in figure.axes[14].get_yticklabels()
            ]
            return ()

        with TemporaryDirectory() as directory:
            with patch.object(
                plotting,
                "save_publication_figure",
                side_effect=capture_ticks,
            ):
                plotting.draw_sound_speed_coupling_grid(
                    bundle,
                    Path(directory) / "reference_aware_peak_ratio",
                    quantity="peak_height_ratio",
                    include_trend=False,
                )

        np.testing.assert_allclose(
            captured["bottom_colorbar_ticks"],
            np.asarray(
                [
                    lower,
                    0.5 * (lower + 1.0),
                    1.0,
                    2.0,
                    4.0,
                    6.0,
                ]
            ),
        )
        self.assertEqual(
            captured["top_colorbar_labels"],
            ["0.996", "0.998", "1", "1.002", "1.004"],
        )
        self.assertEqual(
            captured["bottom_colorbar_labels"],
            ["2.9e-05", "0.5", "1", "2", "4", "6"],
        )

    def test_sound_speed_grid_uses_rs_in_visible_labels(self) -> None:
        panel = np.asarray([[-1.0, 0.5], [0.25, 1.0]], dtype=float)
        values = np.tile(panel, (3, 4, 1, 1))
        bundle = {
            "ratios": np.asarray([0.3, 1.0, 3.0]),
            "nus": np.asarray([0.1, 0.3, 0.5, 1.0]),
            "lambda3": np.asarray([-0.2, 0.2]),
            "lambda4": np.asarray([0.0, 0.2]),
            "log10_beta_ratio": values,
            "fixed_power_maxima": np.max(values, axis=(-2, -1)),
        }
        captured: dict[str, list[str]] = {}

        def capture_labels(
            figure: object,
            output_path: Path | str,
        ) -> tuple[Path, ...]:
            del output_path
            axes = figure.axes
            captured["xlabels"] = [axis.get_xlabel() for axis in axes]
            captured["ylabels"] = [axis.get_ylabel() for axis in axes]
            return ()

        with TemporaryDirectory() as directory:
            with patch.object(
                plotting,
                "save_publication_figure",
                side_effect=capture_labels,
            ):
                plotting.draw_sound_speed_coupling_grid(
                    bundle,
                    Path(directory) / "sound_speed_grid",
                    quantity="log10_beta_ratio",
                )

        visible_labels = captured["xlabels"] + captured["ylabels"]
        row_labels = [
            label for label in captured["ylabels"] if r"$\lambda_4$" in label
        ]
        self.assertEqual(len(row_labels), 3)
        self.assertTrue(all(r"$r_s=" in label for label in row_labels))
        self.assertIn(r"$r_s$", captured["xlabels"])
        self.assertFalse(any(r"$r=" in label for label in visible_labels))
        self.assertFalse(
            any(r"c_\phi/c_\chi" in label for label in visible_labels)
        )
        self.assertFalse(any(r"c_\sigma" in label for label in visible_labels))

    def test_shared_ratio_legends_use_rs(self) -> None:
        import matplotlib.pyplot as plt

        ratios = np.asarray([0.3, 1.0, 3.0])
        nus = np.asarray([0.1, 0.3])
        values = np.ones((3, 2))
        uncertainties = np.full((3, 2), 0.01)
        figure, axes = plt.subplots(1, 2)
        try:
            ratio_handles = plotting._plot_ratio_series(
                axes[0], ratios, nus, values
            )
            coefficient_handles = plotting._plot_coefficient_series(
                axes[1], ratios, nus, values, uncertainties
            )
            labels = [handle.get_label() for handle in ratio_handles]
            labels.extend(handle.get_label() for handle in coefficient_handles)
        finally:
            plt.close(figure)

        self.assertTrue(all(label.startswith(r"$r_s=") for label in labels))
        self.assertFalse(any(label.startswith(r"$r=") for label in labels))


if __name__ == "__main__":
    unittest.main()
