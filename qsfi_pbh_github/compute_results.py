#!/usr/bin/env python3
"""Compute every dataset needed by the final QSFI-PBH figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from qsfi_pbh import backend, production
from qsfi_pbh.abundance import (
    DELTA_C,
    GAMMA_COLLAPSE,
    QNG_CUT_DEFINITION,
    QNG_DEFINITION,
)
from qsfi_pbh.analysis import TARGET_BETA
from qsfi_pbh.config import ProductionConfig
from qsfi_pbh.contributions import fixed_abundance_contributions
from qsfi_pbh.figure_data import write_figure_data
from qsfi_pbh.reduced_parameters import hatted_record_from_barred
from qsfi_pbh.reporting import (
    REPORT_RELATIVE_PATH,
    format_manuscript_numerical_results,
    write_and_print_report,
)
from qsfi_pbh.sound_speed import SOUND_SPEED_CASES
from qsfi_pbh.spectrum import density_variance_response


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_DIR / "figure_data"
DEFAULT_CACHE_ROOT = PACKAGE_DIR / ".cache"
DEFAULT_FIGURE_ROOT = PACKAGE_DIR / "figures"

COEFFICIENT_KEYS = (
    "C3_hat",
    "C4_contact_hat",
    "C4_exchange_hat",
)
COEFFICIENT_ERROR_KEYS = tuple(
    f"{name}_standard_error" for name in COEFFICIENT_KEYS
)


def production_stage_names() -> tuple[str, ...]:
    """Return the public production stages for manuscript outputs."""

    return (
        "r_0p3",
        "r_1",
        "r_3",
        "manuscript_outputs",
    )


def _common_metadata(
    config: ProductionConfig,
    *,
    kind: str,
    description: str,
) -> dict[str, object]:
    metadata = config.metadata()
    variance_response = density_variance_response(
        float(config.lognormal_delta),
        float(backend.K_PEAK_R),
    )
    reference_area = (
        np.sqrt(2.0 * np.pi)
        * float(config.lognormal_delta)
        * float(config.reference_peak_height)
    )
    metadata.update(
        {
            "kind": kind,
            "description": description,
            "formula_version": backend.FORMULA_VERSION,
            "power_normalization": "area_in_dlnk",
            "amplitude_scan_prescription": "fixed_reduced_nonlinearity_parameters",
            "fixed_cumulants_during_peak_height_inversion": True,
            "target_beta": float(TARGET_BETA),
            "variance_response": float(variance_response),
            "C2_variance_response": float(variance_response),
            "variance_relation": "sigma2 = A_zeta * C2",
            "delta_c": float(DELTA_C),
            "gamma_collapse": float(GAMMA_COLLAPSE),
            "qng_definition": QNG_DEFINITION,
            "qng_cut_definition": QNG_CUT_DEFINITION,
            "reference_power_area": float(reference_area),
            "Pzeta_NG_over_Pzeta_G_definition": (
                "P_zeta_NG(k_*) / P_zeta_G(k_*) = "
                "A_zeta_NG / A_zeta_G for fixed lognormal width"
            ),
        }
    )
    return metadata


def _case_metadata(
    case: Mapping[str, object],
    config: ProductionConfig,
    *,
    kind: str,
    description: str,
) -> dict[str, object]:
    sound_speed = case["sound_speed"]
    metadata = _common_metadata(
        config,
        kind=kind,
        description=description,
    )
    metadata.update(
        {
            "ratio": float(sound_speed.ratio),
            "c_phi": float(sound_speed.c_phi),
            "c_sigma": float(sound_speed.c_sigma),
            "qng_definition": QNG_DEFINITION,
            "qng_cut_definition": QNG_CUT_DEFINITION,
        }
    )
    return metadata


def _case_arrays(
    case: Mapping[str, object],
    *,
    value_name: str,
    values: object,
) -> dict[str, np.ndarray]:
    arrays = {
        "nu_tilde": np.asarray(case["nus"], dtype=float),
        value_name: np.asarray(values, dtype=float),
        "Q_NG": np.asarray(case["QNG"], dtype=float),
        "S3": np.asarray(case["S3"], dtype=float),
        "S4": np.asarray(case["S4"], dtype=float),
    }
    arrays["lambda3_over_H"] = np.asarray(
        case["lambda3_over_H"], dtype=float
    )
    arrays["lambda4"] = np.asarray(case["lambda4"], dtype=float)
    return arrays


def export_r3_contribution_dataset(
    *,
    case: Mapping[str, object],
    config: ProductionConfig,
    output_root: Path | str,
) -> Path:
    """Write the r=3 cumulant contributions at fixed-abundance minima."""

    arrays = _case_arrays(
        case,
        value_name="Pzeta_NG_over_Pzeta_G",
        values=case["Pzeta_NG_over_Pzeta_G"],
    )
    return write_figure_data(
        Path(output_root) / "manuscript" / "r3_cumulant_contributions.npz",
        metadata=_case_metadata(
            case,
            config,
            kind="r3_cumulant_contributions",
            description=(
                "Bispectrum and trispectrum contributions at the r=3 "
                "fixed-abundance minima."
            ),
        ),
        arrays=fixed_abundance_contributions(
            arrays,
            qng_max=float(config.qng_max),
        ),
    )


def _coefficient_arrays(
    cases: Sequence[Mapping[str, object]],
    config: ProductionConfig,
) -> dict[str, np.ndarray]:
    ratios = np.asarray(
        [float(case["sound_speed"].ratio) for case in cases],
        dtype=float,
    )
    nus = np.asarray(cases[0]["nus"], dtype=float)
    arrays: dict[str, np.ndarray] = {
        "sound_speed_ratio": ratios,
        "nu_tilde": nus,
    }
    for key in COEFFICIENT_KEYS + COEFFICIENT_ERROR_KEYS:
        values = np.empty((len(cases), nus.size), dtype=float)
        for case_index, case in enumerate(cases):
            records = case["coefficient_records"]
            for nu_index, nu in enumerate(nus):
                converted = hatted_record_from_barred(
                    records[str(float(nu))],
                    reference_area=(
                        np.sqrt(2.0 * np.pi)
                        * float(config.lognormal_delta)
                        * float(config.reference_peak_height)
                    ),
                    lambda2_over_h=float(config.lambda2_over_h),
                )
                values[case_index, nu_index] = float(converted[key])
        arrays[key] = values
    return arrays


def export_manuscript_datasets(
    *,
    cases: Sequence[Mapping[str, object]],
    config: ProductionConfig,
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Write the four manuscript datasets and their numerical-results report."""

    if len(cases) != 3:
        raise ValueError("Exactly three sound-speed cases are required.")

    root = Path(output_root)
    ratios = np.asarray(
        [float(case["sound_speed"].ratio) for case in cases], dtype=float
    )
    expected_ratios = np.asarray((0.3, 1.0, 3.0), dtype=float)
    if not np.array_equal(ratios, expected_ratios):
        raise ValueError(
            "Sound-speed cases must be ordered as r_s=[0.3, 1.0, 3.0] "
            f"with case 2 at r_s=3; found {ratios.tolist()}."
        )
    nus = np.asarray(cases[0]["nus"], dtype=float)
    lambda3 = np.asarray(cases[0]["lambda3_over_H"], dtype=float)
    lambda4 = np.asarray(cases[0]["lambda4"], dtype=float)
    expected_grid_shape = (4, lambda4.size, lambda3.size)
    log10_beta_grids: list[np.ndarray] = []
    peak_height_grids: list[np.ndarray] = []
    s3_grids: list[np.ndarray] = []
    s4_grids: list[np.ndarray] = []
    qng_grids: list[np.ndarray] = []
    for case_index, case in enumerate(cases):
        for field, expected_axis in (
            ("nus", nus),
            ("lambda3_over_H", lambda3),
            ("lambda4", lambda4),
        ):
            axis = np.asarray(case[field], dtype=float)
            if not np.array_equal(axis, expected_axis):
                raise ValueError(
                    f"case {case_index} has mismatched {field} axis."
                )
        for field, grids in (
            ("log10_beta_ratio", log10_beta_grids),
            ("Pzeta_NG_over_Pzeta_G", peak_height_grids),
            ("S3", s3_grids),
            ("S4", s4_grids),
            ("QNG", qng_grids),
        ):
            values = np.asarray(case[field], dtype=float)
            if values.shape != expected_grid_shape:
                raise ValueError(
                    f"case {case_index} {field} must have shape "
                    f"{expected_grid_shape}; found {values.shape}."
                )
            grids.append(values)
    coefficients = _coefficient_arrays(cases, config)
    log10_beta_ratio = np.stack(log10_beta_grids)
    peak_height_ratio = np.stack(peak_height_grids)
    s3 = np.stack(s3_grids)
    s4 = np.stack(s4_grids)
    qng = np.stack(qng_grids)
    common_grid = {
        "sound_speed_ratio": ratios,
        "nu_tilde": nus,
        "lambda3_over_H": lambda3,
        "lambda4": lambda4,
    }

    report = format_manuscript_numerical_results(
        sound_speed_ratio=ratios,
        nu_tilde=nus,
        lambda3_over_H=lambda3,
        lambda4=lambda4,
        log10_beta_ratio=log10_beta_ratio,
        peak_height_ratio=peak_height_ratio,
        S3=s3,
        S4=s4,
        Q_NG=qng,
        coefficients=coefficients,
    )

    written: list[Path] = [
        write_figure_data(
            root / "manuscript" / "hatted_cumulant_coefficients.npz",
            metadata=_common_metadata(
                config,
                kind="hatted_cumulant_coefficients",
                description=(
                    "Amplitude-independent projected cumulant coefficients "
                    "and QMC errors."
                ),
            ),
            arrays=coefficients,
        ),
        export_r3_contribution_dataset(
            case=cases[2],
            config=config,
            output_root=root,
        ),
    ]
    for filename, kind, value_name, values, description in (
        (
            "sound_speed_log10_beta_ratio_full",
            "sound_speed_log10_beta_ratio_full",
            "log10_beta_ratio",
            log10_beta_ratio,
            "Three-ratio fixed-power abundance comparison.",
        ),
        (
            "sound_speed_comparison_full",
            "sound_speed_comparison_full",
            "Pzeta_NG_over_Pzeta_G",
            peak_height_ratio,
            "Three-ratio fixed-abundance peak-height comparison.",
        ),
    ):
        arrays = {**common_grid, value_name: values}
        written.append(
            write_figure_data(
                root / "manuscript" / f"{filename}.npz",
                metadata=_common_metadata(
                    config,
                    kind=kind,
                    description=description,
                ),
                arrays=arrays,
            )
        )

    write_and_print_report(root / REPORT_RELATIVE_PATH, report)
    return tuple(written)


def compute_sound_speed_case(
    sound_speed: object,
    *,
    config: ProductionConfig,
    cache_root: Path,
) -> dict[str, object]:
    """Compute one independent sound-speed case."""

    print(
        "[compute] "
        f"r={sound_speed.ratio:g}, "
        f"c_phi={sound_speed.c_phi:g}, "
        f"c_sigma={sound_speed.c_sigma:g}"
    )
    lambda3, lambda4 = config.coupling_axes()
    case = production._run_case(
        sound_speed,
        output_root=cache_root / "staging",
        cache_root=cache_root,
        lambda3=lambda3,
        lambda4=lambda4,
    )
    case["sound_speed"] = sound_speed
    return case


def validate_production_precision(
    cases: Sequence[Mapping[str, object]],
    config: ProductionConfig,
) -> None:
    """Refuse to label incomplete QMC estimates as manuscript production."""

    rejected: list[str] = []
    for case in cases:
        ratio = float(case["sound_speed"].ratio)
        records = case["coefficient_records"]
        for nu, record in records.items():
            if record.get("sampling_precision_accepted") is not True:
                rejected.append(f"r={ratio:g}, nu_tilde={nu}")
    if rejected:
        raise RuntimeError(
            "Production QMC precision was not accepted for: "
            + "; ".join(rejected)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute all data used by the QSFI-PBH manuscript figures."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    return parser.parse_args()


def main() -> tuple[Path, ...]:
    from make_figures import generate_figures

    args = parse_args()
    config = ProductionConfig()
    cases: list[dict[str, object]] = []
    for stage_index, (stage_name, sound_speed) in enumerate(
        zip(production_stage_names()[:3], SOUND_SPEED_CASES),
        start=1,
    ):
        print(f"[stage {stage_index}/4] {stage_name}: compute and validate")
        case = compute_sound_speed_case(
            sound_speed,
            config=config,
            cache_root=args.cache_root,
        )
        validate_production_precision((case,), config)
        cases.append(case)

    print("[stage 4/4] manuscript_outputs: export and render")
    written = export_manuscript_datasets(
        cases=cases,
        config=config,
        output_root=args.output_root,
    )
    figures = generate_figures(
        data_root=args.output_root,
        output_root=args.figure_root,
    )
    print(
        f"[saved] {len(written)} figure datasets in {args.output_root}; "
        f"1 report; {len(figures)} figure files in {args.figure_root}"
    )
    return written


if __name__ == "__main__":
    main()
