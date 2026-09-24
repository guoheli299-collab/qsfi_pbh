"""Vectorized fields for the complete-shape non-Gaussian parameter scan."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from . import abundance


def fixed_area_fields(
    coefficient: Mapping[str, float],
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    *,
    fixed_area: float,
    reference_area: float,
    variance_response: float,
    qng_max: float,
) -> dict[str, np.ndarray]:
    """Return fixed-reduced-parameter cumulants at one trial power area."""
    l3, l4 = np.meshgrid(
        np.asarray(lambda3, dtype=float),
        np.asarray(lambda4, dtype=float),
    )
    if not math.isfinite(float(reference_area)) or reference_area <= 0.0:
        raise ValueError("reference_area must be positive and finite.")
    s3 = float(coefficient["S3_per_lambda3_over_H"]) * l3
    s4_contact = (
        float(coefficient["S4_per_lambda4_contact"])
        * l4
    )
    s4_exchange = (
        float(coefficient["S4_per_lambda3_over_H_squared_exchange"])
        * l3**2
    )
    s4 = s4_contact + s4_exchange
    qng = abundance.qng_from_cumulants(s3, s4)
    log10_beta_ratio = (
        abundance.log_beta_ratio_from_cumulants(
            fixed_area,
            s3,
            s4,
            variance_response=variance_response,
        )
        / math.log(10.0)
    )
    valid = np.isfinite(log10_beta_ratio) & abundance.qng_is_valid(
        qng,
        qng_max=float(qng_max),
    )
    return {
        "s3": s3,
        "s4_contact": s4_contact,
        "s4_exchange": s4_exchange,
        "s4": s4,
        "qng": qng,
        "log10_beta_ratio": np.where(valid, log10_beta_ratio, np.nan),
        "valid": valid,
    }


def fixed_cumulant_log_target_residual(
    area: np.ndarray,
    s3: np.ndarray | float,
    s4: np.ndarray | float,
    *,
    log_target: float,
    variance_response: float,
) -> np.ndarray:
    """Return log(beta_NG)-log(target) with S3 and S4 held fixed."""

    area_values = np.asarray(area, dtype=float)
    s3_values = np.asarray(s3, dtype=float)
    s4_values = np.asarray(s4, dtype=float)
    if not math.isfinite(float(log_target)):
        raise ValueError("log_target must be finite.")
    if not math.isfinite(float(variance_response)) or variance_response <= 0.0:
        raise ValueError("variance_response must be positive and finite.")
    if np.any(~np.isfinite(area_values)) or np.any(area_values <= 0.0):
        raise ValueError("area must be positive and finite.")
    if np.any(~np.isfinite(s3_values)) or np.any(~np.isfinite(s4_values)):
        raise ValueError("S3 and S4 must be finite.")
    area_values, s3_values, s4_values = np.broadcast_arrays(
        area_values,
        s3_values,
        s4_values,
    )
    delta_c = abundance.DELTA_C
    sigma = np.sqrt(area_values * float(variance_response))
    nu_threshold = delta_c / sigma
    log_gaussian = (
        math.log(abundance.GAMMA_COLLAPSE)
        - 0.5 * math.log(2.0 * math.pi)
        - np.log(nu_threshold)
        - 0.5 * nu_threshold**2
    )
    correction = (
        nu_threshold**2
        * delta_c
        * (s3_values / 6.0 + delta_c * s4_values / 24.0)
    )
    return log_gaussian + correction - float(log_target)


def required_peak_height_fields(
    *,
    s3: np.ndarray,
    s4: np.ndarray,
    qng: np.ndarray,
    target_beta: float,
    variance_response: float,
    qng_max: float,
) -> dict[str, np.ndarray | float]:
    """Solve fixed-beta peak heights without rescaling the cumulants."""

    if not math.isfinite(float(target_beta)) or not 0.0 < target_beta < 1.0:
        raise ValueError("target_beta must satisfy 0 < target_beta < 1.")
    if not math.isfinite(float(qng_max)) or qng_max <= 0.0:
        raise ValueError("qng_max must be positive and finite.")
    s3_values, s4_values, qng_values = np.broadcast_arrays(
        np.asarray(s3, dtype=float),
        np.asarray(s4, dtype=float),
        np.asarray(qng, dtype=float),
    )
    finite = (
        np.isfinite(s3_values)
        & np.isfinite(s4_values)
        & np.isfinite(qng_values)
    )
    perturbative = finite & abundance.qng_is_valid(
        qng_values,
        qng_max=float(qng_max),
    )
    gaussian_required_area = abundance.bisect_for_target_beta(
        float(target_beta),
        variance_response=float(variance_response),
    )
    log_target = math.log(float(target_beta))
    lo = np.full(s3_values.shape, gaussian_required_area, dtype=float)
    hi = np.full(s3_values.shape, gaussian_required_area, dtype=float)
    center = fixed_cumulant_log_target_residual(
        lo,
        np.where(finite, s3_values, 0.0),
        np.where(finite, s4_values, 0.0),
        log_target=log_target,
        variance_response=float(variance_response),
    )

    downward = perturbative & (center >= 0.0)
    active = downward.copy()
    lower_floor = np.finfo(float).tiny * 16.0
    for _ in range(256):
        trial = np.maximum(lo * 0.5, lower_floor)
        residual = fixed_cumulant_log_target_residual(
            trial,
            np.where(finite, s3_values, 0.0),
            np.where(finite, s4_values, 0.0),
            log_target=log_target,
            variance_response=float(variance_response),
        )
        lo = np.where(active, trial, lo)
        active &= (residual > 0.0) & (trial > lower_floor)
        if not np.any(active):
            break

    active = perturbative & ~downward
    for _ in range(256):
        trial = np.minimum(hi * 1.5, 1.0)
        residual = fixed_cumulant_log_target_residual(
            trial,
            np.where(finite, s3_values, 0.0),
            np.where(finite, s4_values, 0.0),
            log_target=log_target,
            variance_response=float(variance_response),
        )
        hi = np.where(active, trial, hi)
        active &= (residual < 0.0) & (trial < 1.0)
        if not np.any(active):
            break

    flo = fixed_cumulant_log_target_residual(
        lo,
        np.where(finite, s3_values, 0.0),
        np.where(finite, s4_values, 0.0),
        log_target=log_target,
        variance_response=float(variance_response),
    )
    fhi = fixed_cumulant_log_target_residual(
        hi,
        np.where(finite, s3_values, 0.0),
        np.where(finite, s4_values, 0.0),
        log_target=log_target,
        variance_response=float(variance_response),
    )
    bracketed = perturbative & (flo <= 0.0) & (fhi >= 0.0)
    for _ in range(180):
        mid = np.sqrt(lo * hi)
        residual = fixed_cumulant_log_target_residual(
            mid,
            np.where(finite, s3_values, 0.0),
            np.where(finite, s4_values, 0.0),
            log_target=log_target,
            variance_response=float(variance_response),
        )
        move_lo = bracketed & (residual < 0.0)
        move_hi = bracketed & ~move_lo
        lo = np.where(move_lo, mid, lo)
        hi = np.where(move_hi, mid, hi)

    required_area = np.sqrt(lo * hi)
    ratio = required_area / gaussian_required_area
    valid = bracketed & np.isfinite(required_area) & np.isfinite(ratio)
    return {
        "required_area": np.where(valid, required_area, np.nan),
        "Pzeta_NG_over_Pzeta_G": np.where(valid, ratio, np.nan),
        "s3": s3_values.copy(),
        "s4": s4_values.copy(),
        "qng": qng_values.copy(),
        "valid": valid,
        "gaussian_required_area": float(gaussian_required_area),
    }


def required_power_fields(
    coefficient: Mapping[str, float],
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    *,
    reference_area: float,
    target_beta: float,
    variance_response: float,
    qng_max: float,
) -> dict[str, np.ndarray | float]:
    """Solve fixed abundance while holding the reduced cumulants fixed."""

    fixed = fixed_area_fields(
        coefficient,
        lambda3,
        lambda4,
        fixed_area=reference_area,
        reference_area=reference_area,
        variance_response=variance_response,
        qng_max=qng_max,
    )
    solved = required_peak_height_fields(
        s3=fixed["s3"],
        s4=fixed["s4"],
        qng=fixed["qng"],
        target_beta=target_beta,
        variance_response=variance_response,
        qng_max=qng_max,
    )
    solved["s4_contact"] = fixed["s4_contact"]
    solved["s4_exchange"] = fixed["s4_exchange"]
    return solved


def optimum_record(
    fields: Mapping[str, np.ndarray],
    objective_key: str,
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    *,
    maximize: bool,
) -> dict[str, float | int]:
    """Extract a reproducible optimum and its cumulant decomposition."""
    lambda3_values = np.asarray(lambda3)
    lambda4_values = np.asarray(lambda4)
    if lambda3_values.ndim != 1 or lambda4_values.ndim != 1:
        raise ValueError(
            "lambda3 and lambda4 must be one-dimensional."
        )

    objective = np.asarray(fields[objective_key], dtype=float)
    expected_shape = (lambda4_values.size, lambda3_values.size)
    if objective.shape != expected_shape:
        raise ValueError(
            f"{objective_key} must have shape {expected_shape}; "
            f"got {objective.shape}."
        )
    field_arrays: dict[str, np.ndarray] = {}
    for key in (
        "valid",
        "s3",
        "s4_contact",
        "s4_exchange",
        "qng",
    ):
        array = np.asarray(fields[key])
        if array.shape != objective.shape:
            raise ValueError(
                f"{key} must have shape {objective.shape}; "
                f"got {array.shape}."
            )
        field_arrays[key] = array

    finite_valid = (
        np.isfinite(objective)
        & field_arrays["valid"].astype(bool, copy=False)
    )
    valid_count = int(np.count_nonzero(finite_valid))
    if valid_count == 0:
        raise RuntimeError(
            f"No valid points remain for {objective_key}."
        )

    ranking = np.where(
        finite_valid,
        objective,
        -np.inf if maximize else np.inf,
    )
    flat_index = int(
        np.argmax(ranking) if maximize else np.argmin(ranking)
    )
    lambda4_index, lambda3_index = np.unravel_index(
        flat_index,
        objective.shape,
    )
    contact = float(
        field_arrays["s4_contact"][
            lambda4_index,
            lambda3_index,
        ]
    )
    exchange = float(
        field_arrays["s4_exchange"][
            lambda4_index,
            lambda3_index,
        ]
    )
    return {
        "objective": float(
            objective[lambda4_index, lambda3_index]
        ),
        "lambda3_over_H": float(
            lambda3_values[lambda3_index]
        ),
        "lambda4": float(lambda4_values[lambda4_index]),
        "s3": float(
            field_arrays["s3"][
                lambda4_index,
                lambda3_index,
            ]
        ),
        "s4_contact": contact,
        "s4_exchange": exchange,
        "s4": contact + exchange,
        "qng": float(
            field_arrays["qng"][
                lambda4_index,
                lambda3_index,
            ]
        ),
        "valid_count": valid_count,
        "excluded_count": int(finite_valid.size - valid_count),
    }
