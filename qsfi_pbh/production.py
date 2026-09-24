#!/usr/bin/env python3
"""Run the complete QSFI-PBH calculation for three sound-speed ratios.

The physical benchmarks are

``(r, c_phi, c_sigma) = (0.3, 0.3, 1), (1, 1, 1), (3, 1, 1/3)``.

All entry points use the single manuscript production resolution.
"""
from __future__ import annotations

import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from . import backend as fig2_common
from . import abundance
from . import analysis as enhancement
from .config import ProductionConfig
from .sound_speed import SOUND_SPEED_CASES, SoundSpeedConfiguration


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_DIR / ".cache" / "staging"
CACHE_ROOT = PROJECT_DIR / ".cache"


def _ratio_label(ratio: float) -> str:
    return f"r_{float(ratio):.12g}".replace("-", "m").replace(".", "p")


def case_paths(
    output_root: Path,
    cache_root: Path,
    sound_speed: SoundSpeedConfiguration,
) -> dict[str, Path]:
    """Return independent figure and cache directories for one benchmark."""

    label = _ratio_label(sound_speed.ratio)
    return {
        "output_dir": Path(output_root) / label,
        "cache_dir": Path(cache_root) / label,
    }


@contextmanager
def _case_environment(
    sound_speed: SoundSpeedConfiguration,
    output_dir: Path,
    cache_dir: Path,
) -> Iterator[None]:
    """Temporarily point the existing figure pipeline at one isolated case."""

    names = (
        "OUTDIR",
        "CACHE_DIR",
        "SUMMARY_PATH",
        "DATA_PATH",
        "DEFAULT_SOUND_SPEED",
    )
    previous = {name: getattr(fig2_common, name) for name in names}
    fig2_common.OUTDIR = output_dir
    fig2_common.CACHE_DIR = cache_dir
    fig2_common.SUMMARY_PATH = output_dir / "qsfi_pbh_cumulant_heatmaps_summary.json"
    fig2_common.DATA_PATH = output_dir / "qsfi_pbh_cumulant_heatmaps_data.npz"
    fig2_common.DEFAULT_SOUND_SPEED = sound_speed
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(fig2_common, name, value)


def _finite_power_of_ten(exponent: float) -> float | None:
    if not math.isfinite(exponent) or exponent > 307.0:
        return None
    return float(10.0**exponent)


def build_case_result(
    sound_speed: SoundSpeedConfiguration,
    analysis: Mapping[str, object],
    *,
    source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Combine scientific metadata, optima, and diagnostic NG ratios."""

    summary = analysis["summary"]
    if not isinstance(summary, Mapping):
        raise TypeError("analysis['summary'] must be a mapping.")
    per_nu_source = summary["per_nu"]
    if not isinstance(per_nu_source, Mapping):
        raise TypeError("analysis summary per_nu must be a mapping.")

    per_nu: dict[str, object] = {}
    for nu in np.asarray(analysis["nus"], dtype=float):
        key = str(float(nu))
        source = per_nu_source[key]
        if not isinstance(source, Mapping):
            raise TypeError(f"analysis record {key!r} must be a mapping.")
        fixed_source = source.get("fixed_peak_height_optimum")
        if fixed_source is None:
            raise ValueError("fixed_peak_height_optimum is required.")
        fixed_optimum = dict(fixed_source)
        abundance_optimum = dict(source["fixed_abundance_optimum"])
        s3 = float(abundance_optimum["s3"])
        s4 = float(abundance_optimum["s4"])
        balance = None if s3 == 0.0 else float(abs(s4) / (8.0 * abs(s3)))
        balance_status = (
            "undefined_zero_S3_at_optimum"
            if balance is None
            else "finite"
        )
        log10_enhancement = float(fixed_optimum["objective"])
        per_nu[key] = {
            "coefficients": dict(source["coefficients"]),
            "fixed_peak_height_optimum": fixed_optimum,
            "fixed_abundance_optimum": abundance_optimum,
            "maximum_log10_beta_enhancement": log10_enhancement,
            "maximum_beta_enhancement": _finite_power_of_ten(log10_enhancement),
            "minimum_Pzeta_NG_over_Pzeta_G": float(
                abundance_optimum["objective"]
            ),
            "abs_S4_over_8_abs_S3": balance,
            "abs_S4_over_8_abs_S3_status": balance_status,
            "abs_contact_fraction_of_S4": (
                None
                if s4 == 0.0
                else float(abs(float(abundance_optimum["s4_contact"])) / abs(s4))
            ),
        }

    return {
        "sound_speed": {
            "ratio": sound_speed.ratio,
            "c_phi": sound_speed.c_phi,
            "c_sigma": sound_speed.c_sigma,
            "i_epsilon": sound_speed.i_epsilon,
        },
        "source_metadata": dict(source_metadata),
        "analysis_metadata": dict(summary["metadata"]),
        "nu_tilde": [float(value) for value in np.asarray(analysis["nus"], dtype=float)],
        "per_nu": per_nu,
        "_arrays": {
            "lambda3_over_H": np.asarray(analysis["lambda3"], dtype=float),
            "lambda4": np.asarray(analysis["lambda4"], dtype=float),
            "log10_beta_ratio": np.asarray(analysis["log10_beta_ratio"], dtype=float),
            "Pzeta_NG_over_Pzeta_G": np.asarray(
                analysis["Pzeta_NG_over_Pzeta_G"],
                dtype=float,
            ),
        },
    }


def _public_case(case_result: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in case_result.items()
        if key != "_arrays"
    }


def _validate_combined_arrays(
    case_results: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(case_results) != 3:
        raise ValueError("Exactly three sound-speed cases are required.")
    reference_nus = np.asarray(case_results[0]["nu_tilde"], dtype=float)
    first_arrays = case_results[0]["_arrays"]
    if not isinstance(first_arrays, Mapping):
        raise TypeError("Case arrays must be a mapping.")
    lambda3 = np.asarray(first_arrays["lambda3_over_H"], dtype=float)
    lambda4 = np.asarray(first_arrays["lambda4"], dtype=float)
    required: list[np.ndarray] = []
    beta: list[np.ndarray] = []
    for case in case_results:
        nus = np.asarray(case["nu_tilde"], dtype=float)
        arrays = case["_arrays"]
        if not isinstance(arrays, Mapping):
            raise TypeError("Case arrays must be a mapping.")
        if not np.array_equal(nus, reference_nus):
            raise ValueError("All sound-speed cases must use identical nu_tilde values.")
        if not np.array_equal(
            np.asarray(arrays["lambda3_over_H"], dtype=float),
            lambda3,
        ):
            raise ValueError("All sound-speed cases must use the same lambda3 grid.")
        if not np.array_equal(np.asarray(arrays["lambda4"], dtype=float), lambda4):
            raise ValueError("All sound-speed cases must use the same lambda4 grid.")
        required.append(
            np.asarray(
                arrays["Pzeta_NG_over_Pzeta_G"],
                dtype=float,
            )
        )
        beta.append(np.asarray(arrays["log10_beta_ratio"], dtype=float))
    return (
        reference_nus,
        lambda3,
        lambda4,
        np.asarray(required, dtype=float),
        np.asarray(beta, dtype=float),
    )


def comparison_layout_spec(nu_count: int) -> dict[str, object]:
    """Return the journal-figure grid contract for the comparison plot."""

    count = int(nu_count)
    if count < 1:
        raise ValueError("nu_count must be positive.")
    return {
        "grid_columns": count + 1,
        "heatmap_columns": tuple(range(count)),
        "colorbar_column": count,
        "trend_columns": (0, count),
        "width_ratios": tuple([1.0] * count + [0.06]),
    }


def _draw_comparison_figure(
    case_results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    nus, lambda3, lambda4, required, _ = _validate_combined_arrays(case_results)
    finite = required[np.isfinite(required)]
    if finite.size == 0:
        raise RuntimeError("No finite required peak-height ratios are available.")

    fig2_common.configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    layout = comparison_layout_spec(int(nus.size))
    figure = plt.figure(figsize=(3.0 * nus.size, 8.5))
    grid = GridSpec(
        4,
        int(layout["grid_columns"]),
        figure=figure,
        height_ratios=(1.0, 1.0, 1.0, 0.85),
        width_ratios=layout["width_ratios"],
    )
    norm = fig2_common.heatmap_norm(finite)
    xgrid, ygrid = np.meshgrid(lambda3, lambda4)
    heat_axes = []
    last_mesh = None
    for row, case in enumerate(case_results):
        sound_speed = case["sound_speed"]
        if not isinstance(sound_speed, Mapping):
            raise TypeError("sound_speed must be a mapping.")
        per_nu = case["per_nu"]
        if not isinstance(per_nu, Mapping):
            raise TypeError("per_nu must be a mapping.")
        for column, nu in enumerate(nus):
            axis = figure.add_subplot(grid[row, column])
            heat_axes.append(axis)
            panel = np.ma.masked_invalid(required[row, column])
            last_mesh = axis.pcolormesh(
                xgrid,
                ygrid,
                panel,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
                rasterized=True,
            )
            optimum = per_nu[str(float(nu))]["fixed_abundance_optimum"]
            axis.plot(
                optimum["lambda3_over_H"],
                optimum["lambda4"],
                marker="*",
                color="black",
                linestyle="none",
                markersize=5.5,
            )
            if row == 0:
                axis.set_title(rf"$\widetilde{{\nu}}={nu:g}$")
            if row == 2:
                axis.set_xlabel(r"$\lambda_3/H$")
            if column == 0:
                axis.set_ylabel(rf"$r={float(sound_speed['ratio']):g}$" + "\n" + r"$\lambda_4$")
            axis.tick_params(direction="in", top=True, right=True, labelsize=7)

    if last_mesh is None:
        raise RuntimeError("No sound-speed comparison panels were drawn.")
    colorbar_axis = figure.add_subplot(grid[0:3, int(layout["colorbar_column"])])
    colorbar = figure.colorbar(last_mesh, cax=colorbar_axis)
    colorbar.set_label(
        r"$\mathcal{P}_{\zeta,\rm NG}(k_*)/"
        r"\mathcal{P}_{\zeta,\rm G}(k_*)$"
    )

    trend_start, trend_stop = layout["trend_columns"]
    trend_axis = figure.add_subplot(grid[3, int(trend_start):int(trend_stop)])
    ratios = np.asarray(
        [float(case["sound_speed"]["ratio"]) for case in case_results],
        dtype=float,
    )
    for nu in nus:
        minima = [
            float(
                case["per_nu"][str(float(nu))][
                    "minimum_Pzeta_NG_over_Pzeta_G"
                ]
            )
            for case in case_results
        ]
        trend_axis.plot(ratios, minima, marker="o", linewidth=1.2, label=rf"$\widetilde{{\nu}}={nu:g}$")
    trend_axis.axhline(1.0, color="0.45", linewidth=0.7)
    trend_axis.set_xscale("log")
    trend_axis.set_xticks(ratios, [f"{value:g}" for value in ratios])
    trend_axis.set_xlabel(r"$r=c_\phi/c_\sigma$")
    trend_axis.set_ylabel(
        r"$\min[\mathcal{P}_{\zeta,\rm NG}(k_*)/"
        r"\mathcal{P}_{\zeta,\rm G}(k_*)]$"
    )
    trend_axis.grid(alpha=0.2)
    trend_axis.legend(ncol=len(nus), fontsize=7, frameon=False)

    figure.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.07, hspace=0.38, wspace=0.25)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def save_comparison_outputs(
    case_results: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    production_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Write the shared comparison PDF, arrays, and self-describing manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    nus, lambda3, lambda4, required, beta = _validate_combined_arrays(case_results)
    public_cases = [_public_case(case) for case in case_results]
    manifest = {
        "metadata": {
            **dict(production_metadata),
            "nu_tilde": [float(value) for value in nus],
            "lambda2_over_H": fig2_common.LAMBDA2_OVER_H,
            "lambda3_over_H_range": [float(lambda3[0]), float(lambda3[-1])],
            "lambda4_range": [float(lambda4[0]), float(lambda4[-1])],
            "lambda3_scan_points": int(lambda3.size),
            "lambda4_scan_points": int(lambda4.size),
            "lognormal_delta": fig2_common.LOGNORMAL_DELTA,
            "reference_peak_height": fig2_common.REFERENCE_PEAK_HEIGHT,
            "reference_power_area": fig2_common.POWER_AREA,
            "target_beta": enhancement.TARGET_BETA,
            "qng_max": enhancement.QNG_MAX,
            "qng_definition": abundance.QNG_DEFINITION,
            "qng_cut_definition": abundance.QNG_CUT_DEFINITION,
            "delta_c": abundance.DELTA_C,
            "gamma_collapse": abundance.GAMMA_COLLAPSE,
            "Pzeta_NG_over_Pzeta_G_definition": (
                "P_zeta_NG(k_*) / P_zeta_G(k_*) = "
                "A_zeta_NG / A_zeta_G for fixed Delta"
            ),
            "k_peak_r": fig2_common.K_PEAK_R,
            "sound_speed_definition": "r = c_phi / c_sigma",
            "complete_shapes": "bispectrum + contact trispectrum + exchange trispectrum",
        },
        "cases": public_cases,
    }

    _draw_comparison_figure(case_results, output_dir / "sound_speed_comparison.pdf")
    np.savez_compressed(
        output_dir / "sound_speed_comparison_data.npz",
        sound_speed_ratios=np.asarray(
            [case["sound_speed"]["ratio"] for case in case_results],
            dtype=float,
        ),
        c_phi=np.asarray(
            [case["sound_speed"]["c_phi"] for case in case_results],
            dtype=float,
        ),
        c_sigma=np.asarray(
            [case["sound_speed"]["c_sigma"] for case in case_results],
            dtype=float,
        ),
        nus=nus,
        lambda3_over_H=lambda3,
        lambda4=lambda4,
        Pzeta_NG_over_Pzeta_G=required,
        log10_beta_ratio=beta,
    )

    final_path = output_dir / "sound_speed_comparison_summary.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(final_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return manifest


def _run_case(
    sound_speed: SoundSpeedConfiguration,
    *,
    output_root: Path,
    cache_root: Path,
    lambda3: np.ndarray | None = None,
    lambda4: np.ndarray | None = None,
) -> dict[str, object]:
    paths = case_paths(output_root, cache_root, sound_speed)
    output_dir = paths["output_dir"]
    cache_dir = paths["cache_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with _case_environment(sound_speed, output_dir, cache_dir):
        fig2_common.ensure_cumulant_data(force=None)
        with open(fig2_common.SUMMARY_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload["metadata"]
        records = payload["coefficients"]
        with np.load(fig2_common.DATA_PATH) as data:
            nus = np.asarray(data["nus"], dtype=float)
            backend_lambda3 = np.asarray(
                data["lambda3_over_H"], dtype=float
            )
            backend_lambda4 = np.asarray(data["lambda4"], dtype=float)

    lambda3_values = (
        backend_lambda3
        if lambda3 is None
        else np.asarray(lambda3, dtype=float)
    )
    lambda4_values = (
        backend_lambda4
        if lambda4 is None
        else np.asarray(lambda4, dtype=float)
    )

    analysis = enhancement.build_analysis(
        records,
        lambda3=lambda3_values,
        lambda4=lambda4_values,
        reference_area=float(metadata["power_area"]),
        fixed_area=enhancement.FIXED_POWER_AREA,
        target_beta=enhancement.TARGET_BETA,
        variance_response=float(metadata["variance_response"]),
        qng_max=enhancement.QNG_MAX,
    )
    return {
        "summary": build_case_result(
            sound_speed,
            analysis,
            source_metadata=metadata,
        ),
        "source_metadata": metadata,
        "coefficient_records": records,
        "nus": nus,
        "lambda3_over_H": lambda3_values,
        "lambda4": lambda4_values,
        "S3": np.asarray(analysis["S3"], dtype=float),
        "S3_standard_error": np.asarray(
            analysis["S3_standard_error"],
            dtype=float,
        ),
        "S4": np.asarray(analysis["S4"], dtype=float),
        "S4_standard_error": np.asarray(
            analysis["S4_standard_error"],
            dtype=float,
        ),
        "QNG": np.asarray(analysis["QNG"], dtype=float),
        "log10_beta_ratio": np.asarray(
            analysis["log10_beta_ratio"],
            dtype=float,
        ),
        "Pzeta_NG_over_Pzeta_G": np.asarray(
            analysis["Pzeta_NG_over_Pzeta_G"],
            dtype=float,
        ),
    }


def main() -> dict[str, object]:
    grid_config = ProductionConfig()
    production_metadata = grid_config.metadata()
    lambda3, lambda4 = grid_config.coupling_axes()
    case_results = []
    for sound_speed in SOUND_SPEED_CASES:
        print(
            "[sound-speed case] "
            f"r={sound_speed.ratio:g}, "
            f"c_phi={sound_speed.c_phi:g}, "
            f"c_sigma={sound_speed.c_sigma:g}"
        )
        case_results.append(
            _run_case(
                sound_speed,
                output_root=OUTPUT_ROOT,
                cache_root=CACHE_ROOT,
                lambda3=lambda3,
                lambda4=lambda4,
            )
        )
    manifest = save_comparison_outputs(
        case_results,
        OUTPUT_ROOT,
        production_metadata=production_metadata,
    )
    print(f"[saved] sound-speed comparison outputs in {OUTPUT_ROOT}")
    return manifest


if __name__ == "__main__":
    main()
