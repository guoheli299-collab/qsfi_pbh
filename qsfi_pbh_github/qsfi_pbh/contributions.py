"""Fixed-abundance bispectrum and trispectrum contribution diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np



def reference_contribution_ratios(
    cases: Sequence[Mapping[str, object]],
    *,
    reference_lambda3_over_H: float,
    reference_lambda4: float,
) -> dict[str, np.ndarray]:
    """Extract signed reference-point ratios for several sound-speed cases."""

    if not cases:
        raise ValueError("At least one sound-speed case is required.")
    reference_nus = np.asarray(cases[0]["nus"], dtype=float)
    ratios: list[float] = []
    s3_rows: list[np.ndarray] = []
    s4_rows: list[np.ndarray] = []
    for case in cases:
        nus = np.asarray(case["nus"], dtype=float)
        if not np.allclose(nus, reference_nus, rtol=0.0, atol=1.0e-12):
            raise ValueError("All sound-speed cases must use the same mass axis.")
        lambda3 = np.asarray(case["lambda3_over_H"], dtype=float)
        lambda4 = np.asarray(case["lambda4"], dtype=float)
        column_matches = np.flatnonzero(
            np.isclose(
                lambda3,
                reference_lambda3_over_H,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        row_matches = np.flatnonzero(
            np.isclose(lambda4, reference_lambda4, rtol=0.0, atol=1.0e-12)
        )
        if column_matches.size != 1 or row_matches.size != 1:
            raise ValueError("A reference coupling is absent or duplicated.")
        s3_grid = np.asarray(case["S3"], dtype=float)
        s4_grid = np.asarray(case["S4"], dtype=float)
        expected = (nus.size, lambda4.size, lambda3.size)
        if s3_grid.shape != expected or s4_grid.shape != expected:
            raise ValueError(f"Reference grids must have shape {expected}.")
        column = int(column_matches[0])
        row = int(row_matches[0])
        s3_rows.append(s3_grid[:, row, column])
        s4_rows.append(s4_grid[:, row, column])
        sound_speed = case["sound_speed"]
        ratios.append(float(sound_speed.ratio))
    s3_values = np.asarray(s3_rows, dtype=float)
    s4_values = np.asarray(s4_rows, dtype=float)
    if np.any(s3_values == 0.0):
        raise ValueError("The signed reference ratio is undefined where S3 is zero.")
    trispectrum = s4_values / 8.0
    return {
        "sound_speed_ratio": np.asarray(ratios, dtype=float),
        "nu_tilde": reference_nus.copy(),
        "S3": s3_values,
        "S4_over_8": trispectrum,
        "Q_NG": s3_values + trispectrum,
        "signed_S4_over_8_over_S3": trispectrum / s3_values,
    }


def fixed_abundance_contributions(
    arrays: Mapping[str, object],
    *,
    qng_max: float,
) -> dict[str, np.ndarray]:
    """Select each mass panel's allowed peak-height minimum and decompose Q_NG."""

    nus = np.asarray(arrays["nu_tilde"], dtype=float)
    lambda3 = np.asarray(arrays["lambda3_over_H"], dtype=float)
    lambda4 = np.asarray(arrays["lambda4"], dtype=float)
    if nus.ndim != 1 or lambda3.ndim != 1 or lambda4.ndim != 1:
        raise ValueError("nu_tilde, lambda3_over_H, and lambda4 must be one-dimensional.")

    expected = (nus.size, lambda4.size, lambda3.size)
    fields = {
        name: np.asarray(arrays[name], dtype=float)
        for name in ("Pzeta_NG_over_Pzeta_G", "Q_NG", "S3", "S4")
    }
    for name, values in fields.items():
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, found {values.shape}.")

    selected_indices: list[tuple[int, int]] = []
    for nu_index in range(nus.size):
        ratio_panel = fields["Pzeta_NG_over_Pzeta_G"][nu_index]
        qng_panel = fields["Q_NG"][nu_index]
        valid = (
            np.isfinite(ratio_panel)
            & np.isfinite(qng_panel)
            & (qng_panel <= float(qng_max))
        )
        if not np.any(valid):
            raise ValueError(f"Mass panel nu_tilde={nus[nu_index]:g} has no valid point.")
        eligible = np.where(valid, ratio_panel, np.nan)
        selected_indices.append(np.unravel_index(np.nanargmin(eligible), eligible.shape))

    lambda4_indices = np.asarray([index[0] for index in selected_indices], dtype=int)
    lambda3_indices = np.asarray([index[1] for index in selected_indices], dtype=int)
    nu_indices = np.arange(nus.size, dtype=int)

    selected = {
        name: values[nu_indices, lambda4_indices, lambda3_indices]
        for name, values in fields.items()
    }
    s4_over_8 = selected["S4"] / 8.0
    if not np.allclose(
        selected["Q_NG"],
        selected["S3"] + s4_over_8,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("Selected Q_NG values do not equal S3 + S4/8.")
    dominance = np.divide(
        np.abs(s4_over_8),
        np.abs(selected["S3"]),
        out=np.full_like(s4_over_8, np.inf),
        where=selected["S3"] != 0.0,
    )
    return {
        "nu_tilde": nus.copy(),
        "lambda3_over_H_at_minimum": lambda3[lambda3_indices],
        "lambda4_at_minimum": lambda4[lambda4_indices],
        "Pzeta_NG_over_Pzeta_G": selected["Pzeta_NG_over_Pzeta_G"],
        "S3": selected["S3"],
        "S4": selected["S4"],
        "S4_over_8": s4_over_8,
        "Q_NG": selected["Q_NG"],
        "abs_S4_over_8_over_S3": dominance,
    }
