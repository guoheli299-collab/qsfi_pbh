#!/usr/bin/env python3
"""Area-normalized lognormal power spectrum and finite-width leg weights."""
from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

DEFAULT_DELTA = 0.5
DEFAULT_K_PEAK_R = math.sqrt(2.0)
POWER_NORMALIZATION = "area_in_dlnk"
_LOG_MIN_FLOAT = math.log(np.finfo(float).tiny)


def _validate_shape(delta: float, k_peak_r: float) -> None:
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("lognormal width delta must be positive and finite.")
    if not math.isfinite(k_peak_r) or k_peak_r <= 0.0:
        raise ValueError("k_peak_r must be positive and finite.")


def _positive_array(x: ArrayLike, name: str) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain only positive finite values.")
    return values


def _positive_finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return scalar


def area_from_peak_height(
    peak_height: float,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Return the integrated lognormal area for a specified peak height."""

    height = _positive_finite_scalar(peak_height, "peak_height")
    width = _positive_finite_scalar(delta, "delta")
    return math.sqrt(2.0 * math.pi) * width * height


def peak_height_from_area(
    area: float,
    delta: float = DEFAULT_DELTA,
) -> float:
    """Return the lognormal peak height for a specified integrated area."""

    integrated_area = _positive_finite_scalar(area, "area")
    width = _positive_finite_scalar(delta, "delta")
    return integrated_area / (math.sqrt(2.0 * math.pi) * width)


def log_unit_area_profile(
    x: ArrayLike,
    delta: float = DEFAULT_DELTA,
    k_peak_r: float = DEFAULT_K_PEAK_R,
) -> np.ndarray:
    """Return ``log(u_delta)`` where ``integral u_delta dln(x) = 1``."""
    _validate_shape(delta, k_peak_r)
    values = _positive_array(x, "x")
    log_ratio = np.log(values / k_peak_r)
    return -0.5 * (log_ratio / delta) ** 2 - math.log(math.sqrt(2.0 * math.pi) * delta)


def unit_area_profile(
    x: ArrayLike,
    delta: float = DEFAULT_DELTA,
    k_peak_r: float = DEFAULT_K_PEAK_R,
) -> np.ndarray:
    """Return the unit-area lognormal profile as a function of ``x=kR``."""
    log_values = log_unit_area_profile(x, delta, k_peak_r)
    return np.exp(np.maximum(log_values, _LOG_MIN_FLOAT))


def power_spectrum(
    x: ArrayLike,
    area: float,
    delta: float = DEFAULT_DELTA,
    k_peak_r: float = DEFAULT_K_PEAK_R,
) -> np.ndarray:
    """Return ``P_zeta(x)=area*u_delta(x)`` with area measured in ``dln k``."""
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("power-spectrum area must be positive and finite.")
    return area * unit_area_profile(x, delta, k_peak_r)


def density_variance_response(
    delta: float = DEFAULT_DELTA,
    k_peak_r: float = DEFAULT_K_PEAK_R,
    *,
    quadrature_order: int = 80,
) -> float:
    r"""Return ``C2`` in ``sigma^2=A*C2`` for ``W=exp[-(kR)^2/2]``.

    Gauss--Hermite nodes evaluate the expectation value over the Gaussian
    random variable ``ln(x/k_peak_r)`` exactly in the lognormal measure.
    """
    _validate_shape(delta, k_peak_r)
    if int(quadrature_order) < 8:
        raise ValueError("quadrature_order must be at least 8.")
    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_order))
    x = k_peak_r * np.exp(math.sqrt(2.0) * delta * nodes)
    kernel = (16.0 / 81.0) * x**4 * np.exp(-(x**2))
    response = float(np.dot(weights, kernel) / math.sqrt(math.pi))
    if not math.isfinite(response) or response <= 0.0:
        raise FloatingPointError("density-variance response is not positive and finite.")
    return response


def density_variance(
    area: float,
    delta: float = DEFAULT_DELTA,
    k_peak_r: float = DEFAULT_K_PEAK_R,
    *,
    quadrature_order: int = 80,
) -> float:
    """Return the smoothed density variance for an area-normalized spectrum."""
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("power-spectrum area must be positive and finite.")
    return area * density_variance_response(
        delta,
        k_peak_r,
        quadrature_order=quadrature_order,
    )
