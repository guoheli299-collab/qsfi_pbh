"""Amplitude-independent QSFI cumulants at fixed reduced parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .collapse import qng_from_cumulants


BARRED_TO_HATTED = {
    "S3_per_lambda3_over_H": "C3_hat",
    "S3_per_lambda3_over_H_standard_error": "C3_hat_standard_error",
    "S4_per_lambda4_contact": "C4_contact_hat",
    "S4_per_lambda4_contact_standard_error": (
        "C4_contact_hat_standard_error"
    ),
    "S4_per_lambda3_over_H_squared_exchange": "C4_exchange_hat",
    "S4_per_lambda3_over_H_squared_exchange_standard_error": (
        "C4_exchange_hat_standard_error"
    ),
}


@dataclass(frozen=True)
class ReducedParameters:
    """Reduced nonlinearity parameters fixed during an amplitude scan."""

    f_nl: np.ndarray
    tau_nl_ci: np.ndarray
    tau_nl_se: np.ndarray


@dataclass(frozen=True)
class DensityCumulants:
    """Third- and fourth-order density cumulants and propagated errors."""

    s3: np.ndarray
    s4_contact: np.ndarray
    s4_exchange: np.ndarray
    s4: np.ndarray
    qng: np.ndarray
    s3_standard_error: np.ndarray
    s4_contact_standard_error: np.ndarray
    s4_exchange_standard_error: np.ndarray
    s4_standard_error: np.ndarray


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _finite_array(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def reference_area(*, delta: float, peak_height: float) -> float:
    """Return ``A_zeta`` for a fixed-width lognormal peak."""

    width = _positive_finite("delta", delta)
    peak = _positive_finite("peak_height", peak_height)
    return math.sqrt(2.0 * math.pi) * width * peak


def hatted_record_from_barred(
    record: Mapping[str, object],
    *,
    reference_area: float,
    lambda2_over_h: float,
) -> dict[str, np.ndarray | float]:
    """Convert reference-amplitude barred coefficients to hatted ones."""

    area = _positive_finite("reference_area", reference_area)
    mixing = _positive_finite("lambda2_over_h", lambda2_over_h)
    missing = [key for key in BARRED_TO_HATTED if key not in record]
    if missing:
        raise ValueError(f"Coefficient record is missing fields: {missing}")
    cubic_factor = math.sqrt(area) / mixing**3
    quartic_factor = area / mixing**4
    converted: dict[str, np.ndarray | float] = {}
    for source, destination in BARRED_TO_HATTED.items():
        values = _finite_array(source, record[source])
        if source.endswith("_standard_error") and np.any(values < 0.0):
            raise ValueError(f"{source} contains a negative standard error.")
        factor = cubic_factor if source.startswith("S3_") else quartic_factor
        result = values * factor
        converted[destination] = float(result) if result.ndim == 0 else result
    return converted


def reduced_parameters_from_reference_couplings(
    *,
    lambda3_0_over_h: object,
    lambda4_0: object,
    reference_area: float,
    lambda2_over_h: float,
) -> ReducedParameters:
    """Map reference raw couplings to fixed reduced nonlinearity parameters."""

    area = _positive_finite("reference_area", reference_area)
    mixing = _positive_finite("lambda2_over_h", lambda2_over_h)
    lambda3 = _finite_array("lambda3_0_over_h", lambda3_0_over_h)
    lambda4 = _finite_array("lambda4_0", lambda4_0)
    try:
        lambda3, lambda4 = np.broadcast_arrays(lambda3, lambda4)
    except ValueError as error:
        raise ValueError(
            "lambda3_0_over_h and lambda4_0 must be broadcast-compatible."
        ) from error
    return ReducedParameters(
        f_nl=mixing**3 * lambda3 / math.sqrt(area),
        tau_nl_ci=mixing**4 * lambda4 / area,
        tau_nl_se=mixing**4 * lambda3**2 / area,
    )


def cumulants_from_reduced_parameters(
    hatted: Mapping[str, object],
    reduced: ReducedParameters,
) -> DensityCumulants:
    """Evaluate fixed S3, S4 and Q_NG from hatted coefficients."""

    required = (
        "C3_hat",
        "C3_hat_standard_error",
        "C4_contact_hat",
        "C4_contact_hat_standard_error",
        "C4_exchange_hat",
        "C4_exchange_hat_standard_error",
    )
    missing = [key for key in required if key not in hatted]
    if missing:
        raise ValueError(f"Hatted coefficient record is missing fields: {missing}")
    values = {key: _finite_array(key, hatted[key]) for key in required}
    for key in required:
        if key.endswith("_standard_error") and np.any(values[key] < 0.0):
            raise ValueError(f"{key} contains a negative standard error.")

    s3 = values["C3_hat"] * reduced.f_nl
    s4_contact = values["C4_contact_hat"] * reduced.tau_nl_ci
    s4_exchange = values["C4_exchange_hat"] * reduced.tau_nl_se
    s4 = s4_contact + s4_exchange
    s3_error = values["C3_hat_standard_error"] * np.abs(reduced.f_nl)
    contact_error = values["C4_contact_hat_standard_error"] * np.abs(
        reduced.tau_nl_ci
    )
    exchange_error = values["C4_exchange_hat_standard_error"] * np.abs(
        reduced.tau_nl_se
    )
    return DensityCumulants(
        s3=np.asarray(s3, dtype=float),
        s4_contact=np.asarray(s4_contact, dtype=float),
        s4_exchange=np.asarray(s4_exchange, dtype=float),
        s4=np.asarray(s4, dtype=float),
        qng=qng_from_cumulants(s3, s4),
        s3_standard_error=np.asarray(s3_error, dtype=float),
        s4_contact_standard_error=np.asarray(contact_error, dtype=float),
        s4_exchange_standard_error=np.asarray(exchange_error, dtype=float),
        s4_standard_error=np.asarray(
            np.sqrt(contact_error**2 + exchange_error**2),
            dtype=float,
        ),
    )
