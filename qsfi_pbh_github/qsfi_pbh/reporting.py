"""Deterministic text records for manuscript numerical results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
import tempfile
from typing import TextIO

import numpy as np


REPORT_RELATIVE_PATH = Path("manuscript_numerical_results.txt")
COEFFICIENT_FIELDS = (
    "C3_hat",
    "C3_hat_standard_error",
    "C4_contact_hat",
    "C4_contact_hat_standard_error",
    "C4_exchange_hat",
    "C4_exchange_hat_standard_error",
)


def _axis(
    name: str,
    values: object,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Convert and validate one of the report's ordered scan axes."""

    try:
        axis = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional axis.") from error
    if axis.ndim != 1 or axis.size < 2 or np.any(~np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite one-dimensional axis.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    if expected_size is not None and axis.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values.")
    return axis


def _as_float_array(name: str, values: object) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error


def _extrema_lines(
    *,
    ratios: np.ndarray,
    nus: np.ndarray,
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    values: object,
    maximize: bool,
    associated: Mapping[str, object],
) -> list[str]:
    grids = _as_float_array("extremum grid", values)
    expected = (ratios.size, nus.size, lambda4.size, lambda3.size)
    if grids.shape != expected:
        raise ValueError(f"Extremum grids must have shape {expected}.")
    associated_grids: dict[str, np.ndarray] = {}
    for name in ("S3", "S4", "Q_NG"):
        if name not in associated:
            raise ValueError(f"Missing extremum-associated grid {name!r}.")
        grid = _as_float_array(name, associated[name])
        if grid.shape != expected:
            raise ValueError(f"{name} grids must have shape {expected}.")
        if np.any(~np.isfinite(grid)):
            raise ValueError(f"{name} grids must be finite.")
        associated_grids[name] = grid
    lines = [
        "\t".join(
            (
                "r_s",
                "nu_tilde",
                "extremum",
                "lambda3_over_H",
                "lambda4",
                "S3",
                "S4",
                "Q_NG",
            )
        )
    ]
    for ratio_index, ratio in enumerate(ratios):
        for nu_index, nu in enumerate(nus):
            panel = grids[ratio_index, nu_index]
            finite = np.isfinite(panel)
            if not np.any(finite):
                raise ValueError(
                    f"Panel r_s={ratio:g}, nu_tilde={nu:g} has no finite values."
                )
            eligible = np.where(finite, panel, np.nan)
            flat_index = int(
                np.nanargmax(eligible) if maximize else np.nanargmin(eligible)
            )
            lambda4_index, lambda3_index = np.unravel_index(
                flat_index, panel.shape
            )
            selected = tuple(
                associated_grids[name][
                    ratio_index,
                    nu_index,
                    lambda4_index,
                    lambda3_index,
                ]
                for name in ("S3", "S4", "Q_NG")
            )
            lines.append(
                f"{ratio:g}\t{nu:g}\t{panel[lambda4_index, lambda3_index]:.8g}"
                f"\t{lambda3[lambda3_index]:+.3f}\t{lambda4[lambda4_index]:+.3f}"
                + "".join(f"\t{value:.8g}" for value in selected)
            )
    return lines


def format_manuscript_numerical_results(
    *,
    sound_speed_ratio: object,
    nu_tilde: object,
    lambda3_over_H: object,
    lambda4: object,
    log10_beta_ratio: object,
    peak_height_ratio: object,
    S3: object,
    S4: object,
    Q_NG: object,
    coefficients: Mapping[str, object],
) -> str:
    """Format all manuscript extrema and projected coefficients deterministically."""

    ratios = _axis("sound_speed_ratio", sound_speed_ratio, expected_size=3)
    nus = _axis("nu_tilde", nu_tilde, expected_size=4)
    lambda3 = _axis("lambda3_over_H", lambda3_over_H)
    quartic = _axis("lambda4", lambda4)
    associated = {"S3": S3, "S4": S4, "Q_NG": Q_NG}

    if not isinstance(coefficients, Mapping):
        raise ValueError("coefficients must provide the required coefficient fields.")
    coefficient_arrays: dict[str, np.ndarray] = {}
    for name in COEFFICIENT_FIELDS:
        if name not in coefficients:
            raise ValueError(f"Missing coefficient field {name!r}.")
        values = _as_float_array(f"coefficient {name}", coefficients[name])
        if values.shape != (3, 4) or np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must be a finite 3-by-4 array.")
        coefficient_arrays[name] = values

    lines = [
        "QSFI-PBH manuscript numerical results",
        f"lambda3_over_H=[{lambda3[0]:+.3f}, {lambda3[-1]:+.3f}] "
        f"({lambda3.size} points)",
        f"lambda4=[{quartic[0]:+.3f}, {quartic[-1]:+.3f}] "
        f"({quartic.size} points)",
        "",
        "[fixed_power_extrema]",
    ]
    lines.extend(
        _extrema_lines(
            ratios=ratios,
            nus=nus,
            lambda3=lambda3,
            lambda4=quartic,
            values=log10_beta_ratio,
            maximize=True,
            associated=associated,
        )
    )
    lines.extend(("", "[fixed_abundance_extrema]"))
    lines.extend(
        _extrema_lines(
            ratios=ratios,
            nus=nus,
            lambda3=lambda3,
            lambda4=quartic,
            values=peak_height_ratio,
            maximize=False,
            associated=associated,
        )
    )
    lines.extend(
        (
            "",
            "[projected_coefficients]",
            "\t".join(("r_s", "nu_tilde", *COEFFICIENT_FIELDS)),
        )
    )
    for ratio_index, ratio in enumerate(ratios):
        for nu_index, nu in enumerate(nus):
            values = "\t".join(
                f"{coefficient_arrays[name][ratio_index, nu_index]:.12e}"
                for name in COEFFICIENT_FIELDS
            )
            lines.append(f"{ratio:g}\t{nu:g}\t{values}")
    return "\n".join(lines) + "\n"


def write_and_print_report(
    path: Path | str,
    report: str,
    *,
    stream: TextIO | None = None,
) -> Path:
    """Atomically write *report*, then emit that exact text to *stream*."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(report)
        Path(temporary_name).replace(destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    (sys.stdout if stream is None else stream).write(report)
    return destination
