"""PBH collapse convention and threshold-derived perturbativity controls."""

from __future__ import annotations

import math
from typing import Union

import numpy as np


ArrayLike = Union[float, np.ndarray]

DELTA_C = 0.5
GAMMA_COLLAPSE = 0.2


def qng_from_cumulants(
    s3: ArrayLike,
    s4: ArrayLike,
    *,
    delta_c: float = DELTA_C,
) -> np.ndarray:
    r"""Return ``Q_NG=S3+delta_c*S4/4`` at fourth cumulant order."""

    threshold = float(delta_c)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("delta_c must be positive and finite.")
    s3_values, s4_values = np.broadcast_arrays(
        np.asarray(s3, dtype=float),
        np.asarray(s4, dtype=float),
    )
    if np.any(~np.isfinite(s3_values)) or np.any(~np.isfinite(s4_values)):
        raise ValueError("S3 and S4 must be finite.")
    return np.asarray(s3_values + threshold * s4_values / 4.0, dtype=float)


def qng_limit(*, delta_c: float = DELTA_C) -> float:
    """Return the one-sided fourth-order high-peak upper bound."""

    threshold = float(delta_c)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("delta_c must be positive and finite.")
    return 3.0 / threshold


QNG_MAX = qng_limit()


def qng_is_valid(
    qng: ArrayLike,
    *,
    qng_max: float = QNG_MAX,
) -> np.ndarray:
    """Return the finite one-sided mask ``Q_NG <= qng_max``."""

    limit = float(qng_max)
    if not math.isfinite(limit):
        raise ValueError("qng_max must be finite.")
    values = np.asarray(qng, dtype=float)
    return np.isfinite(values) & (values <= limit)


QNG_DEFINITION = "Q_NG = S3 + delta_c*S4/4"
QNG_CUT_DEFINITION = f"Q_NG <= {QNG_MAX:g}"
VALID_QNG_FIELD = f"valid_QNG_le_{QNG_MAX:g}"
