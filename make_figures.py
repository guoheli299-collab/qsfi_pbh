#!/usr/bin/env python3
"""Render the three manuscript figures from independent NPZ datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsfi_pbh import plotting
from qsfi_pbh.figure_data import FigureData, load_figure_data


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_DIR / "figure_data"
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR / "figures"

_FIGURE_STEMS = (
    "r3_relative_contribution_bars",
    "sound_speed_comparison_full",
    "sound_speed_log10_beta_ratio_full",
)
EXPECTED_FIGURE_PATHS = tuple(
    Path(f"{stem}.{suffix}")
    for stem in _FIGURE_STEMS
    for suffix in ("pdf", "svg", "png")
)


def _sound_speed_plot_bundle(
    data: FigureData,
    *,
    value_key: str,
) -> dict[str, np.ndarray]:
    """Select one full sound-speed field and its axes for the grid renderer."""

    values = np.asarray(data.arrays[value_key], dtype=float)
    bundle = {
        "ratios": np.asarray(data.arrays["sound_speed_ratio"], dtype=float),
        "nus": np.asarray(data.arrays["nu_tilde"], dtype=float),
        "lambda3": np.asarray(data.arrays["lambda3_over_H"], dtype=float),
        "lambda4": np.asarray(data.arrays["lambda4"], dtype=float),
        value_key: values,
    }
    if value_key == "log10_beta_ratio":
        bundle["fixed_power_maxima"] = np.nanmax(values, axis=(-2, -1))
    elif value_key == "Pzeta_NG_over_Pzeta_G":
        bundle["fixed_abundance_minima"] = np.nanmin(values, axis=(-2, -1))
    else:
        raise ValueError(f"Unsupported sound-speed field: {value_key!r}.")
    return bundle


def generate_manuscript_figures(
    data_root: Path | str,
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Render only the retained contribution and sound-speed manuscript figures."""

    data_dir = Path(data_root)
    figure_dir = Path(output_root)
    contribution_data = load_figure_data(
        data_dir / "manuscript" / "r3_cumulant_contributions.npz",
        expected_kind="r3_cumulant_contributions",
    )
    sound_beta = load_figure_data(
        data_dir / "manuscript" / "sound_speed_log10_beta_ratio_full.npz",
        expected_kind="sound_speed_log10_beta_ratio_full",
    )
    sound_peak = load_figure_data(
        data_dir / "manuscript" / "sound_speed_comparison_full.npz",
        expected_kind="sound_speed_comparison_full",
    )
    outputs: list[Path] = []
    outputs.extend(
        plotting.draw_relative_contribution_bars(
            contribution_data.arrays,
            figure_dir / "r3_relative_contribution_bars",
        )
    )
    outputs.extend(
        plotting.draw_sound_speed_coupling_grid(
            _sound_speed_plot_bundle(sound_beta, value_key="log10_beta_ratio"),
            figure_dir / "sound_speed_log10_beta_ratio_full",
            quantity="log10_beta_ratio",
            include_trend=False,
        )
    )
    outputs.extend(
        plotting.draw_sound_speed_coupling_grid(
            _sound_speed_plot_bundle(
                sound_peak,
                value_key="Pzeta_NG_over_Pzeta_G",
            ),
            figure_dir / "sound_speed_comparison_full",
            quantity="peak_height_ratio",
            include_trend=False,
        )
    )
    return tuple(outputs)


def generate_figures(
    *,
    data_root: Path | str,
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Render the three supported manuscript figures in PDF, SVG, and PNG."""

    figure_dir = Path(output_root)
    outputs = generate_manuscript_figures(data_root, figure_dir)
    expected = {figure_dir / path for path in EXPECTED_FIGURE_PATHS}
    actual = set(outputs)
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RuntimeError(
            f"Figure output contract mismatch; missing={missing}, extra={extra}"
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render three manuscript figures from NPZ data."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> tuple[Path, ...]:
    args = parse_args()
    outputs = generate_figures(
        data_root=args.data_root,
        output_root=args.output_root,
    )
    print(f"[saved] {len(outputs)} figure files in {args.output_root}")
    return outputs


if __name__ == "__main__":
    main()
