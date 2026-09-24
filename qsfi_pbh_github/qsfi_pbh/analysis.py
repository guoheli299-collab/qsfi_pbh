#!/usr/bin/env python3
"""Build the single-case non-Gaussian enhancement analysis."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from . import backend as fig2_common
from . import abundance
from .config import ProductionConfig
from .spectrum import peak_height_from_area
from .enhancement import (
    optimum_record,
    required_peak_height_fields,
)
from .reduced_parameters import (
    cumulants_from_reduced_parameters,
    hatted_record_from_barred,
    reduced_parameters_from_reference_couplings,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "figures"

EXPECTED_NUS = (0.1, 0.3, 0.5, 1.0)
EXPECTED_LAMBDA2_OVER_H = ProductionConfig().lambda2_over_h
EXPECTED_LOGNORMAL_DELTA = 0.5
EXPECTED_K_PEAK_R = math.sqrt(2.0)
FIXED_POWER_AREA = fig2_common.POWER_AREA
TARGET_BETA = 1.0e-15
QNG_MAX = abundance.QNG_MAX

COEFFICIENT_FIELDS = (
    "C3_hat",
    "C3_hat_standard_error",
    "C4_contact_hat",
    "C4_contact_hat_standard_error",
    "C4_exchange_hat",
    "C4_exchange_hat_standard_error",
)


def build_analysis(
    coefficient_records: Mapping[str, Mapping[str, object]],
    *,
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    reference_area: float,
    fixed_area: float,
    target_beta: float,
    variance_response: float,
    qng_max: float,
    lognormal_delta: float = EXPECTED_LOGNORMAL_DELTA,
) -> dict[str, object]:
    """Build both scan objectives from already-computed coefficient records."""
    normalized_records: dict[float, Mapping[str, object]] = {}
    original_keys: dict[float, object] = {}
    for raw_key, record in coefficient_records.items():
        try:
            nu = float(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Mass key {raw_key!r} must be finite numeric."
            ) from exc
        if not math.isfinite(nu):
            raise ValueError(
                f"Mass key {raw_key!r} must be finite numeric."
            )
        if nu in normalized_records:
            raise ValueError(
                "Duplicate numeric mass aliases "
                f"{original_keys[nu]!r} and {raw_key!r}."
            )
        normalized_records[nu] = record
        original_keys[nu] = raw_key
    nus = np.asarray(
        sorted(normalized_records),
        dtype=float,
    )
    fixed_panels: list[np.ndarray] = []
    required_panels: list[np.ndarray] = []
    s3_panels: list[np.ndarray] = []
    s3_error_panels: list[np.ndarray] = []
    s4_panels: list[np.ndarray] = []
    s4_error_panels: list[np.ndarray] = []
    qng_panels: list[np.ndarray] = []
    gaussian_beta_at_fixed_power = float(
        abundance.beta_gaussian(
            fixed_area,
            variance_response=variance_response,
        )
    )
    gaussian_required_power_area = float(
        abundance.bisect_for_target_beta(
            target_beta,
            variance_response=variance_response,
        )
    )
    reference_peak_height = peak_height_from_area(reference_area, lognormal_delta)
    fixed_power_peak_height = peak_height_from_area(fixed_area, lognormal_delta)
    gaussian_required_peak_height = peak_height_from_area(
        gaussian_required_power_area,
        lognormal_delta,
    )
    summary: dict[str, object] = {
        "metadata": {
            "fixed_power_area": float(fixed_area),
            "fixed_power_peak_height": fixed_power_peak_height,
            "target_beta": float(target_beta),
            "reference_area": float(reference_area),
            "reference_peak_height": reference_peak_height,
            "lognormal_delta": float(lognormal_delta),
            "variance_response": float(variance_response),
            "C2_variance_response": float(variance_response),
            "variance_relation": "sigma2 = A_zeta * C2",
            "qng_max": float(qng_max),
            "qng_definition": abundance.QNG_DEFINITION,
            "qng_cut_definition": abundance.QNG_CUT_DEFINITION,
            "gaussian_beta_at_fixed_power": (
                gaussian_beta_at_fixed_power
            ),
            "gaussian_required_power_area": (
                gaussian_required_power_area
            ),
            "gaussian_required_peak_height": (
                gaussian_required_peak_height
            ),
            "gaussian_required_sigma2": (
                gaussian_required_power_area * float(variance_response)
            ),
        },
        "per_nu": {},
    }
    per_nu = summary["per_nu"]
    assert isinstance(per_nu, dict)

    lambda3_values = np.asarray(lambda3, dtype=float)
    lambda4_values = np.asarray(lambda4, dtype=float)
    lambda3_grid, lambda4_grid = np.meshgrid(
        lambda3_values,
        lambda4_values,
    )
    reduced = reduced_parameters_from_reference_couplings(
        lambda3_0_over_h=lambda3_grid,
        lambda4_0=lambda4_grid,
        reference_area=reference_area,
        lambda2_over_h=EXPECTED_LAMBDA2_OVER_H,
    )
    for nu in nus:
        key = str(float(nu))
        coefficient = hatted_record_from_barred(
            normalized_records[float(nu)],
            reference_area=reference_area,
            lambda2_over_h=EXPECTED_LAMBDA2_OVER_H,
        )
        cumulants = cumulants_from_reduced_parameters(
            coefficient,
            reduced,
        )
        fixed_log10_ratio = (
            abundance.log_beta_ratio_from_cumulants(
                fixed_area,
                cumulants.s3,
                cumulants.s4,
                variance_response=variance_response,
            )
            / math.log(10.0)
        )
        fixed_valid = (
            np.isfinite(fixed_log10_ratio)
            & abundance.qng_is_valid(
                cumulants.qng,
                qng_max=float(qng_max),
            )
        )
        fixed = {
            "s3": cumulants.s3,
            "s4_contact": cumulants.s4_contact,
            "s4_exchange": cumulants.s4_exchange,
            "s4": cumulants.s4,
            "qng": cumulants.qng,
            "log10_beta_ratio": np.where(
                fixed_valid,
                fixed_log10_ratio,
                np.nan,
            ),
            "valid": fixed_valid,
        }
        solved = required_peak_height_fields(
            s3=cumulants.s3,
            s4=cumulants.s4,
            qng=cumulants.qng,
            target_beta=target_beta,
            variance_response=variance_response,
            qng_max=qng_max,
        )
        required = {
            **solved,
            "s4_contact": cumulants.s4_contact,
            "s4_exchange": cumulants.s4_exchange,
        }
        fixed_panels.append(fixed["log10_beta_ratio"])
        required_panels.append(required["Pzeta_NG_over_Pzeta_G"])
        s3_panels.append(cumulants.s3)
        s3_error_panels.append(cumulants.s3_standard_error)
        s4_panels.append(cumulants.s4)
        s4_error_panels.append(cumulants.s4_standard_error)
        qng_panels.append(cumulants.qng)
        fixed_peak_height_optimum = optimum_record(
            fixed,
            "log10_beta_ratio",
            lambda3_values,
            lambda4_values,
            maximize=True,
        )
        fixed_abundance_optimum = optimum_record(
            required,
            "Pzeta_NG_over_Pzeta_G",
            lambda3_values,
            lambda4_values,
            maximize=False,
        )
        per_nu[key] = {
            "gaussian_beta_at_fixed_power": (
                gaussian_beta_at_fixed_power
            ),
            "gaussian_required_power_area": (
                gaussian_required_power_area
            ),
            "gaussian_required_peak_height": (
                gaussian_required_peak_height
            ),
            "coefficients": {
                name: float(coefficient[name])
                for name in COEFFICIENT_FIELDS
            },
            "fixed_peak_height_optimum": fixed_peak_height_optimum,
            "fixed_abundance_optimum": fixed_abundance_optimum,
        }

    return {
        "nus": nus,
        "lambda3": lambda3_values,
        "lambda4": lambda4_values,
        "S3": np.asarray(s3_panels, dtype=float),
        "S3_standard_error": np.asarray(s3_error_panels, dtype=float),
        "S4": np.asarray(s4_panels, dtype=float),
        "S4_standard_error": np.asarray(s4_error_panels, dtype=float),
        "QNG": np.asarray(qng_panels, dtype=float),
        "log10_beta_ratio": np.asarray(fixed_panels, dtype=float),
        "Pzeta_NG_over_Pzeta_G": np.asarray(
            required_panels,
            dtype=float,
        ),
        "summary": summary,
    }


def load_coefficient_records(
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Generate/load coefficient data and validate the mass benchmark."""
    configured_nus = tuple(float(nu) for nu in fig2_common.NUS)
    if configured_nus != EXPECTED_NUS:
        raise ValueError(
            f"Expected configured masses {list(EXPECTED_NUS)}, "
            f"found {list(configured_nus)}."
        )
    fig2_common.ensure_cumulant_data(force=None)
    with open(fig2_common.SUMMARY_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata = payload.get("metadata")
    records = payload.get("coefficients")
    if not isinstance(metadata, dict):
        raise ValueError("Cumulant summary metadata must be an object.")
    if not isinstance(records, dict):
        raise ValueError("Cumulant summary coefficients must be an object.")

    found_nus = tuple(sorted(float(key) for key in records))
    if found_nus != EXPECTED_NUS:
        raise ValueError(
            f"Expected masses {list(EXPECTED_NUS)}, "
            f"found {list(found_nus)}."
        )
    expected_scalars = {
        "lambda2_over_H": EXPECTED_LAMBDA2_OVER_H,
        "lognormal_delta": EXPECTED_LOGNORMAL_DELTA,
        "k_peak_r": EXPECTED_K_PEAK_R,
    }
    for name, expected in expected_scalars.items():
        try:
            found = float(metadata[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing or invalid metadata field {name!r}."
            ) from exc
        if not math.isclose(
            found,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ValueError(
                f"Expected {name}={expected}, found {found}."
            )

    validated_records: dict[str, dict[str, object]] = {}
    for nu in EXPECTED_NUS:
        key = str(float(nu))
        record = records.get(key)
        if not isinstance(record, dict):
            raise ValueError(
                f"Coefficient record {key!r} must be an object."
            )
        for field in COEFFICIENT_FIELDS:
            try:
                value = float(record[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing or invalid {field!r} for nu={key}."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite {field!r} for nu={key}."
                )
        validated_records[key] = record
    return validated_records, metadata


def draw_objective(
    values: np.ndarray,
    analysis: Mapping[str, object],
    *,
    output_path: Path,
    colorbar_label: str,
    optimum_name: str,
) -> None:
    """Draw four mass panels using one shared color normalization."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError(
            f"No finite values available for {output_path.name}."
        )

    fig2_common.configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lambda3 = np.asarray(analysis["lambda3"], dtype=float)
    lambda4 = np.asarray(analysis["lambda4"], dtype=float)
    nus = np.asarray(analysis["nus"], dtype=float)
    expected_shape = (nus.size, lambda4.size, lambda3.size)
    if values.shape != expected_shape:
        raise ValueError(
            f"Objective array must have shape {expected_shape}; "
            f"got {values.shape}."
        )
    if nus.size != len(EXPECTED_NUS):
        raise ValueError(
            f"Expected four mass panels; found {nus.size}."
        )

    xgrid, ygrid = np.meshgrid(lambda3, lambda4)
    common_norm = fig2_common.heatmap_norm(finite)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    try:
        per_nu = analysis["summary"]["per_nu"]
        last_mesh = None
        for index, (axis, nu) in enumerate(
            zip(axes.ravel(), nus)
        ):
            last_mesh = axis.pcolormesh(
                xgrid,
                ygrid,
                np.ma.masked_invalid(values[index]),
                shading="auto",
                cmap="RdBu_r",
                norm=common_norm,
                rasterized=True,
            )
            optimum = per_nu[str(float(nu))][optimum_name]
            axis.plot(
                optimum["lambda3_over_H"],
                optimum["lambda4"],
                marker="*",
                color="black",
                markeredgecolor="black",
                linestyle="none",
                markersize=7,
                zorder=5,
            )
            axis.set_title(rf"$\widetilde{{\nu}}={nu:g}$")
            if index // 2 == 1:
                axis.set_xlabel(r"$\lambda_3/H$")
            if index % 2 == 0:
                axis.set_ylabel(r"$\lambda_4$")

        if last_mesh is None:
            raise RuntimeError("No panels were drawn.")
        colorbar = fig.colorbar(
            last_mesh,
            ax=axes.ravel().tolist(),
            fraction=0.035,
            pad=0.03,
        )
        colorbar.set_label(colorbar_label)
        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    finally:
        plt.close(fig)


def save_outputs(
    analysis: Mapping[str, object],
    output_dir: Path,
) -> None:
    """Write the two figures, compressed arrays, and atomic JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_objective(
        np.asarray(analysis["log10_beta_ratio"], dtype=float),
        analysis,
        output_path=(
            output_dir / "ng_enhancement_fixed_peak_height.pdf"
        ),
        colorbar_label=(
            r"$\log_{10}(\beta_{\rm NG}/\beta_{\rm G})$"
        ),
        optimum_name="fixed_peak_height_optimum",
    )
    draw_objective(
        np.asarray(analysis["Pzeta_NG_over_Pzeta_G"], dtype=float),
        analysis,
        output_path=(
            output_dir / "ng_enhancement_required_peak_height.pdf"
        ),
        colorbar_label=(
            r"$\mathcal{P}_{\zeta,\rm NG}(k_*)/"
            r"\mathcal{P}_{\zeta,\rm G}(k_*)$"
        ),
        optimum_name="fixed_abundance_optimum",
    )
    np.savez_compressed(
        output_dir
        / "ng_enhancement_parameter_test_data.npz",
        nus=np.asarray(analysis["nus"], dtype=float),
        lambda3_over_H=np.asarray(
            analysis["lambda3"],
            dtype=float,
        ),
        lambda4=np.asarray(analysis["lambda4"], dtype=float),
        log10_beta_ratio=np.asarray(
            analysis["log10_beta_ratio"],
            dtype=float,
        ),
        Pzeta_NG_over_Pzeta_G=np.asarray(
            analysis["Pzeta_NG_over_Pzeta_G"],
            dtype=float,
        ),
    )

    final_path = (
        output_dir
        / "ng_enhancement_parameter_test_summary.json"
    )
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
            json.dump(
                analysis["summary"],
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
        temporary_path.replace(final_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> dict[str, object]:
    """Run the production non-Gaussian enhancement scan."""
    production_metadata = ProductionConfig().metadata()
    coefficient_records, source_metadata = load_coefficient_records()
    try:
        reference_area = float(source_metadata["power_area"])
        variance_response = float(
            source_metadata["variance_response"]
        )
        lambda3_range = np.asarray(
            source_metadata["lambda3_over_H_range"],
            dtype=float,
        )
        lambda4_range = np.asarray(
            source_metadata["lambda4_range"],
            dtype=float,
        )
        nscan = int(source_metadata["nscan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cumulant metadata lacks valid scan or abundance settings."
        ) from exc
    if (
        lambda3_range.shape != (2,)
        or lambda4_range.shape != (2,)
        or not np.all(np.isfinite(lambda3_range))
        or not np.all(np.isfinite(lambda4_range))
        or nscan < 2
    ):
        raise ValueError(
            "Cumulant scan ranges must be finite pairs and nscan >= 2."
        )

    lambda3 = np.linspace(
        lambda3_range[0],
        lambda3_range[1],
        nscan,
    )
    lambda4 = np.linspace(
        lambda4_range[0],
        lambda4_range[1],
        nscan,
    )
    analysis = build_analysis(
        coefficient_records,
        lambda3=lambda3,
        lambda4=lambda4,
        reference_area=reference_area,
        fixed_area=FIXED_POWER_AREA,
        target_beta=TARGET_BETA,
        variance_response=variance_response,
        qng_max=QNG_MAX,
    )
    analysis_metadata = dict(analysis["summary"]["metadata"])
    analysis["summary"]["metadata"] = {
        **source_metadata,
        **analysis_metadata,
        **production_metadata,
        "nu_tilde": list(EXPECTED_NUS),
        "mass_over_H": [
            math.sqrt(2.25 + nu**2)
            for nu in EXPECTED_NUS
        ],
    }
    save_outputs(analysis, OUTPUT_DIR)
    print(
        "[saved] non-Gaussian enhancement parameter-test "
        f"outputs in {OUTPUT_DIR}"
    )
    return analysis


if __name__ == "__main__":
    main()
