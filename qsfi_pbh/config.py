"""Immutable physical and numerical settings for the reproducible workflow."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .collapse import QNG_MAX


PRODUCTION_CONTRACT_VERSION = "qsfi-pbh-update17-lambda2p38-wide-v1"


@dataclass(frozen=True)
class SoundSpeedCase:
    """One subluminal sound-speed benchmark."""

    ratio: float
    c_phi: float
    c_sigma: float

    def __post_init__(self) -> None:
        values = (self.ratio, self.c_phi, self.c_sigma)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Sound speeds and their ratio must be positive.")
        if self.c_phi > 1.0 or self.c_sigma > 1.0:
            raise ValueError("Individual sound speeds must not exceed one.")
        if not math.isclose(
            self.ratio,
            self.c_phi / self.c_sigma,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError("ratio must equal c_phi / c_sigma.")

    @property
    def label(self) -> str:
        return ratio_label(self.ratio)


def ratio_label(ratio: float) -> str:
    """Return the stable directory label for a supported sound-speed ratio."""

    value = float(ratio)
    if math.isclose(value, 0.3, rel_tol=0.0, abs_tol=1.0e-12):
        return "r_0p3"
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "r_1"
    if math.isclose(value, 3.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "r_3"
    raise ValueError(f"Unsupported production sound-speed ratio: {value:g}")


PRODUCTION_SOUND_SPEED_CASES = (
    SoundSpeedCase(ratio=0.3, c_phi=0.3, c_sigma=1.0),
    SoundSpeedCase(ratio=1.0, c_phi=1.0, c_sigma=1.0),
    SoundSpeedCase(ratio=3.0, c_phi=1.0, c_sigma=1.0 / 3.0),
)


@dataclass(frozen=True)
class ProductionConfig:
    """All settings that affect production data or cache identity."""

    lambda3_points: int = 2001
    lambda4_points: int = 2001
    nus: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    lambda3_min: float = -1.0
    lambda3_max: float = 1.0
    lambda4_min: float = -1.0
    lambda4_max: float = 1.0
    lognormal_delta: float = 0.5
    reference_peak_height: float = 1.0e-2
    k_peak_r: float = math.sqrt(2.0)
    lambda2_over_h: float = 0.38
    qng_max: float = QNG_MAX
    full_momentum_sigma_range: float = 5.0
    full_ratio_floor: float = 1.0e-5
    full_ratio_nodes: int = 128
    wick_pivot: float = 4.0
    wick_depth: float = 16.0
    tail_depth: float = 18.0
    real_step: float = 0.025
    vertical_step: float = 0.025
    nlog: int = 500
    xmin: float = 1.0e-6
    dps: int = 50
    sobol_power_bispectrum: int = 12
    sobol_power_contact: int = 12
    sobol_power_exchange: int = 12
    sobol_replicates: int = 4
    sobol_min_replicates_four_point: int = 6
    sobol_max_replicates_four_point: int = 12
    sobol_relative_error_target_bispectrum: float = 0.01
    sobol_relative_error_target_four_point: float = 0.05
    sobol_max_power_increment_four_point: int = 3
    sobol_seed: int = 1729

    def __post_init__(self) -> None:
        if self.lambda3_points < 3:
            raise ValueError("lambda3_points must be at least three.")
        if self.lambda4_points < 3:
            raise ValueError("lambda4_points must be at least three.")
        if self.lambda3_min >= self.lambda3_max:
            raise ValueError("lambda3 bounds must be increasing.")
        if self.lambda4_min >= self.lambda4_max:
            raise ValueError("lambda4 bounds must be increasing.")

    def coupling_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the true evaluation axes; no interpolation is performed."""

        return (
            np.linspace(
                self.lambda3_min,
                self.lambda3_max,
                self.lambda3_points,
                dtype=float,
            ),
            np.linspace(
                self.lambda4_min,
                self.lambda4_max,
                self.lambda4_points,
                dtype=float,
            ),
        )

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe settings used by every output dataset."""

        return {
            "production_contract_version": PRODUCTION_CONTRACT_VERSION,
            "mode": "production",
            "manuscript_usable": True,
            "lambda3_points": self.lambda3_points,
            "lambda4_points": self.lambda4_points,
            "lambda3_step": (
                (self.lambda3_max - self.lambda3_min)
                / (self.lambda3_points - 1)
            ),
            "lambda4_step": (
                (self.lambda4_max - self.lambda4_min)
                / (self.lambda4_points - 1)
            ),
            "nus": list(self.nus),
            "lambda3_range": [self.lambda3_min, self.lambda3_max],
            "lambda4_range": [self.lambda4_min, self.lambda4_max],
            "lognormal_delta": self.lognormal_delta,
            "reference_peak_height": self.reference_peak_height,
            "k_peak_r": self.k_peak_r,
            "lambda2_over_H": self.lambda2_over_h,
            "qng_max": self.qng_max,
            "full_ratio_nodes": self.full_ratio_nodes,
            "sobol_power_bispectrum": self.sobol_power_bispectrum,
            "sobol_power_contact": self.sobol_power_contact,
            "sobol_power_exchange": self.sobol_power_exchange,
            "sobol_replicates": self.sobol_replicates,
            "sobol_min_replicates_four_point": (
                self.sobol_min_replicates_four_point
            ),
            "sobol_max_replicates_four_point": (
                self.sobol_max_replicates_four_point
            ),
            "sobol_max_power_increment_four_point": (
                self.sobol_max_power_increment_four_point
            ),
            "sobol_seed": self.sobol_seed,
        }
