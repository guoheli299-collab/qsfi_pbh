#!/usr/bin/env python3
"""PBH cumulants from complete QSFI momentum shapes.

The three-point projection is three dimensional after rotational symmetry;
the connected four-point projection is six dimensional. Expensive shape
evaluation is separated from analytic coupling factors so scans in lambda_3
and lambda_4 do not repeat time integrals.
"""
from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from .config import ProductionConfig
from .momentum_shapes import (
    FourMomentumConfiguration,
    FullMomentumShapeEvaluator,
    RatioDomainError,
    build_ratio_function_bundle,
)
from .spectrum import (
    area_from_peak_height,
    density_variance_response,
    power_spectrum,
)
from .sound_speed import SoundSpeedConfiguration

Stage = Literal["bispectrum", "contact", "exchange"]
RadialSampling = Literal["log_uniform", "sqrt_lognormal_importance"]


@dataclass(frozen=True)
class FullMomentumIntegrationSettings:
    """Scientific and numerical settings for the complete PBH projection."""

    sound_speed: SoundSpeedConfiguration = SoundSpeedConfiguration(
        1.0,
        1.0,
        i_epsilon=0.0,
    )
    lognormal_delta: float = 0.5
    k_peak_r: float = math.sqrt(2.0)
    lognormal_sigma_range: float = 5.0
    power_area: float = area_from_peak_height(1.0e-2, 0.5)
    lambda2_over_H: float = ProductionConfig().lambda2_over_h
    ratio_floor: float = 1.0e-5
    ratio_nodes: int = 96
    radial_sampling: RadialSampling = "sqrt_lognormal_importance"
    sobol_power_bispectrum: int = 10
    sobol_power_contact: int = 9
    sobol_power_exchange: int = 8
    sobol_replicates: int = 4
    sobol_min_replicates_four_point: int = 6
    sobol_max_replicates_four_point: int = 12
    sobol_relative_error_target_bispectrum: float = 0.01
    sobol_relative_error_target_four_point: float = 0.05
    sobol_max_power_increment_four_point: int = 3
    sobol_seed: int = 1729

    def __post_init__(self) -> None:
        if not isinstance(self.sound_speed, SoundSpeedConfiguration):
            raise TypeError("sound_speed must be a SoundSpeedConfiguration.")
        positive = {
            "lognormal_delta": self.lognormal_delta,
            "k_peak_r": self.k_peak_r,
            "lognormal_sigma_range": self.lognormal_sigma_range,
            "power_area": self.power_area,
            "lambda2_over_H": self.lambda2_over_H,
            "ratio_floor": self.ratio_floor,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.lambda2_over_H > 0.8:
            raise ValueError("lambda2_over_H must not exceed the active EFT ceiling 0.8.")
        if self.ratio_floor >= 1.0:
            raise ValueError("ratio_floor must be smaller than one.")
        if int(self.ratio_nodes) < 3:
            raise ValueError("ratio_nodes must be at least three.")
        if self.radial_sampling not in ("log_uniform", "sqrt_lognormal_importance"):
            raise ValueError(
                "radial_sampling must be 'log_uniform' or "
                "'sqrt_lognormal_importance'."
            )
        for name in (
            "sobol_power_bispectrum",
            "sobol_power_contact",
            "sobol_power_exchange",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least one.")
        if int(self.sobol_replicates) < 2:
            raise ValueError("sobol_replicates must be at least two for an uncertainty estimate.")
        if int(self.sobol_min_replicates_four_point) < 2:
            raise ValueError("sobol_min_replicates_four_point must be at least two.")
        if int(self.sobol_max_replicates_four_point) < int(
            self.sobol_min_replicates_four_point
        ):
            raise ValueError(
                "sobol_max_replicates_four_point must not be smaller than the minimum."
            )
        for name in (
            "sobol_relative_error_target_bispectrum",
            "sobol_relative_error_target_four_point",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if int(self.sobol_max_power_increment_four_point) < 0:
            raise ValueError("sobol_max_power_increment_four_point must be non-negative.")

    @property
    def k_min(self) -> float:
        return float(self.k_peak_r * math.exp(-self.lognormal_sigma_range * self.lognormal_delta))

    @property
    def k_max(self) -> float:
        return float(self.k_peak_r * math.exp(+self.lognormal_sigma_range * self.lognormal_delta))

    @property
    def log_span(self) -> float:
        return float(math.log(self.k_max / self.k_min))

    def sobol_power(self, stage: Stage) -> int:
        if stage == "bispectrum":
            return int(self.sobol_power_bispectrum)
        if stage == "contact":
            return int(self.sobol_power_contact)
        if stage == "exchange":
            return int(self.sobol_power_exchange)
        raise ValueError(f"Unknown projection stage {stage!r}.")

    def minimum_replicates(self, stage: Stage) -> int:
        if stage == "bispectrum":
            return int(self.sobol_replicates)
        if stage in ("contact", "exchange"):
            return int(self.sobol_min_replicates_four_point)
        raise ValueError(f"Unknown projection stage {stage!r}.")

    def maximum_replicates(self, stage: Stage) -> int:
        if stage == "bispectrum":
            return int(self.sobol_replicates)
        if stage in ("contact", "exchange"):
            return int(self.sobol_max_replicates_four_point)
        raise ValueError(f"Unknown projection stage {stage!r}.")

    def relative_error_target(self, stage: Stage) -> float:
        if stage == "bispectrum":
            return float(self.sobol_relative_error_target_bispectrum)
        if stage in ("contact", "exchange"):
            return float(self.sobol_relative_error_target_four_point)
        raise ValueError(f"Unknown projection stage {stage!r}.")

    def maximum_sobol_power(self, stage: Stage) -> int:
        base = self.sobol_power(stage)
        if stage == "bispectrum":
            return base
        if stage in ("contact", "exchange"):
            return base + int(self.sobol_max_power_increment_four_point)
        raise ValueError(f"Unknown projection stage {stage!r}.")


def _validated_unit_points(points: np.ndarray, dimension: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(f"Sobol points must have shape (n, {dimension}).")
    if np.any(~np.isfinite(points)) or np.any(points < 0.0) or np.any(points >= 1.0):
        raise ValueError("Sobol points must be finite and lie in [0, 1).")
    return points


def _map_radial_points(
    unit: np.ndarray,
    settings: FullMomentumIntegrationSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Map unit coordinates to momenta and return exact ``dk/du`` factors."""
    unit = np.asarray(unit, dtype=float)
    if settings.radial_sampling == "log_uniform":
        momenta = settings.k_min * np.exp(settings.log_span * unit)
        return momenta, settings.log_span * momenta

    sigma_y = math.sqrt(2.0) * settings.lognormal_delta
    z_limit = settings.lognormal_sigma_range / math.sqrt(2.0)
    cdf_min = float(ndtr(-z_limit))
    cdf_span = float(ndtr(z_limit) - cdf_min)
    probabilities = cdf_min + cdf_span * unit
    z = ndtri(probabilities)
    momenta = settings.k_peak_r * np.exp(sigma_y * z)
    density_y = np.exp(-0.5 * z**2) / (
        math.sqrt(2.0 * math.pi) * sigma_y * cdf_span
    )
    return momenta, momenta / density_y


def map_bispectrum_points(
    points: np.ndarray,
    settings: FullMomentumIntegrationSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map unit-cube points to ``(k1,k2,mu12)`` and return ``dk1 dk2 dmu/du``."""
    points = _validated_unit_points(points, 3)
    momenta, radial_jacobian = _map_radial_points(points[:, :2], settings)
    k1 = momenta[:, 0]
    k2 = momenta[:, 1]
    mu = -1.0 + 2.0 * points[:, 2]
    jacobian = 2.0 * radial_jacobian[:, 0] * radial_jacobian[:, 1]
    return k1, k2, mu, jacobian


def map_trispectrum_point(
    point: np.ndarray,
    settings: FullMomentumIntegrationSettings,
) -> tuple[FourMomentumConfiguration, float]:
    """Map one unit-cube point to the reduced six-dimensional four-point geometry."""
    points = _validated_unit_points(np.asarray(point, dtype=float)[None, :], 6)
    unit = points[0]
    momenta, radial_jacobian = _map_radial_points(unit[:3], settings)
    k1, k2, k3 = momenta
    mu12 = -1.0 + 2.0 * unit[3]
    mu13 = -1.0 + 2.0 * unit[4]
    phi3 = 2.0 * math.pi * unit[5]
    configuration = FourMomentumConfiguration.from_reduced_coordinates(
        float(k1),
        float(k2),
        float(k3),
        mu12=float(mu12),
        mu13=float(mu13),
        phi3=float(phi3),
    )
    jacobian = 8.0 * math.pi * float(np.prod(radial_jacobian))
    return configuration, jacobian


def reduced_cumulant_prefactors(
    settings: FullMomentumIntegrationSettings,
) -> tuple[float, float, float]:
    r"""Return exact angular/Fourier prefactors multiplying the raw integrals."""
    sigma2 = settings.power_area * density_variance_response(
        settings.lognormal_delta,
        settings.k_peak_r,
    )
    prefactor3 = (
        2.0
        * (4.0 / 9.0) ** 3
        * settings.lambda2_over_H**3
        / sigma2**2
    )
    prefactor4 = (
        (1.0 / math.pi)
        * (4.0 / 9.0) ** 4
        * settings.lambda2_over_H**4
        / sigma2**3
    )
    return float(prefactor3), float(prefactor4), float(sigma2)


def _window_and_leg_factor(
    momenta: np.ndarray,
    settings: FullMomentumIntegrationSettings,
) -> float:
    momenta = np.asarray(momenta, dtype=float)
    if np.any(~np.isfinite(momenta)) or np.any(momenta <= 0.0):
        return 0.0
    physical_power = power_spectrum(
        momenta,
        settings.power_area,
        settings.lognormal_delta,
        settings.k_peak_r,
    )
    log_leg = 0.5 * float(np.sum(np.log(physical_power)))
    log_weight = log_leg - 0.5 * float(np.sum(momenta**2))
    if log_weight < math.log(np.finfo(float).tiny):
        return 0.0
    return math.exp(log_weight)


def bispectrum_sample_value(
    point: np.ndarray,
    evaluator: FullMomentumShapeEvaluator,
    settings: FullMomentumIntegrationSettings,
) -> float:
    """Return the raw three-dimensional integrand after unit-cube mapping."""
    k1, k2, mu, jacobian = map_bispectrum_points(np.asarray(point)[None, :], settings)
    k3_squared = k1[0] ** 2 + k2[0] ** 2 + 2.0 * k1[0] * k2[0] * mu[0]
    if k3_squared <= 0.0:
        return 0.0
    k3 = math.sqrt(float(k3_squared))
    momenta = np.asarray([k1[0], k2[0], k3])
    spectral = _window_and_leg_factor(momenta, settings)
    if spectral == 0.0:
        return 0.0
    shape = evaluator.bispectrum_shape(float(k1[0]), float(k2[0]), k3)
    result = jacobian[0] * k1[0] ** 2 * k2[0] ** 2 * spectral * shape
    return float(result)


def trispectrum_sample_value(
    point: np.ndarray,
    evaluator: FullMomentumShapeEvaluator,
    settings: FullMomentumIntegrationSettings,
    *,
    stage: Literal["contact", "exchange"],
) -> float:
    """Return one raw six-dimensional contact or exchange integrand sample."""
    configuration, jacobian = map_trispectrum_point(point, settings)
    momenta = configuration.magnitudes
    spectral = _window_and_leg_factor(momenta, settings)
    if spectral == 0.0:
        return 0.0
    if stage == "contact":
        shape = evaluator.contact_shape_per_lambda4(*(float(value) for value in momenta))
    elif stage == "exchange":
        shape = evaluator.exchange_shape_per_lambda3_squared(configuration)
    else:
        raise ValueError("stage must be 'contact' or 'exchange'.")
    radial = momenta[0] * momenta[1] * momenta[2] / momenta[3]
    result = jacobian * radial * configuration.total_magnitude**3 * spectral * shape
    return float(result)


@dataclass(frozen=True)
class StageEstimate:
    """One coupling-independent full-momentum cumulant coefficient estimate."""

    stage: Stage
    raw_integral: float
    raw_standard_error: float
    coefficient: float
    coefficient_standard_error: float
    samples_per_replicate: int
    completed_replicates: int
    sobol_power_used: int
    relative_standard_error: float
    relative_error_target: float
    precision_accepted: bool
    stop_reason: str
    attempted_sobol_powers: tuple[int, ...]
    sampling_history: tuple[dict[str, object], ...]
    excluded_ratio_samples: int
    excluded_ratio_fraction: float
    replicate_integrals: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "raw_integral": self.raw_integral,
            "raw_standard_error": self.raw_standard_error,
            "coefficient": self.coefficient,
            "coefficient_standard_error": self.coefficient_standard_error,
            "samples_per_replicate": self.samples_per_replicate,
            "completed_replicates": self.completed_replicates,
            "sobol_power_used": self.sobol_power_used,
            "relative_standard_error": self.relative_standard_error,
            "relative_error_target": self.relative_error_target,
            "precision_accepted": self.precision_accepted,
            "stop_reason": self.stop_reason,
            "attempted_sobol_powers": list(self.attempted_sobol_powers),
            "sampling_history": list(self.sampling_history),
            "excluded_ratio_samples": self.excluded_ratio_samples,
            "excluded_ratio_fraction": self.excluded_ratio_fraction,
            "replicate_integrals": list(self.replicate_integrals),
        }


def compare_stage_estimates(
    coarse: StageEstimate,
    refined: StageEstimate,
    *,
    relative_tolerance: float = 0.05,
    sigma_multiplier: float = 3.0,
    absolute_floor: float = 1.0e-12,
) -> dict[str, float | bool | str]:
    """Test a refinement shift against tolerance and QMC uncertainty."""
    if coarse.stage != refined.stage:
        raise ValueError("Convergence estimates must refer to the same stage.")
    for name, value in (
        ("relative_tolerance", relative_tolerance),
        ("sigma_multiplier", sigma_multiplier),
        ("absolute_floor", absolute_floor),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    difference = abs(refined.coefficient - coarse.coefficient)
    scale = max(abs(coarse.coefficient), abs(refined.coefficient), absolute_floor)
    combined_standard_error = math.hypot(
        coarse.coefficient_standard_error,
        refined.coefficient_standard_error,
    )
    threshold = max(
        absolute_floor,
        relative_tolerance * scale,
        sigma_multiplier * combined_standard_error,
    )
    sampling_threshold = max(absolute_floor, relative_tolerance * scale)
    statistically_consistent = difference <= threshold
    coarse_sampling_precision_accepted = coarse.coefficient_standard_error <= sampling_threshold
    refined_sampling_precision_accepted = refined.coefficient_standard_error <= sampling_threshold
    sampling_precision_accepted = (
        coarse_sampling_precision_accepted and refined_sampling_precision_accepted
    )
    return {
        "stage": coarse.stage,
        "coarse_coefficient": coarse.coefficient,
        "refined_coefficient": refined.coefficient,
        "absolute_difference": difference,
        "relative_difference": difference / scale,
        "combined_standard_error": combined_standard_error,
        "acceptance_threshold": threshold,
        "sampling_precision_threshold": sampling_threshold,
        "coarse_relative_standard_error": coarse.coefficient_standard_error / scale,
        "refined_relative_standard_error": refined.coefficient_standard_error / scale,
        "statistically_consistent": bool(statistically_consistent),
        "coarse_sampling_precision_accepted": bool(coarse_sampling_precision_accepted),
        "refined_sampling_precision_accepted": bool(refined_sampling_precision_accepted),
        "sampling_precision_accepted": bool(sampling_precision_accepted),
        "accepted": bool(statistically_consistent and sampling_precision_accepted),
    }


def _checkpoint_integration_identity(
    settings: FullMomentumIntegrationSettings,
) -> dict[str, object]:
    """Return settings that affect sample values, excluding stopping controls."""
    identity = asdict(settings)
    for key in (
        "sobol_power_bispectrum",
        "sobol_power_contact",
        "sobol_power_exchange",
        "sobol_replicates",
        "sobol_min_replicates_four_point",
        "sobol_max_replicates_four_point",
        "sobol_relative_error_target_bispectrum",
        "sobol_relative_error_target_four_point",
        "sobol_max_power_increment_four_point",
    ):
        identity.pop(key)
    return identity


def _checkpoint_payload(
    stage: Stage,
    settings: FullMomentumIntegrationSettings,
    evaluator_identity: dict[str, object],
    levels: dict[int, dict[str, list[float] | list[int]]],
) -> dict[str, object]:
    return {
        "format_version": "full_momentum_qsfi_stage_v5",
        "stage": stage,
        "integration_identity": _checkpoint_integration_identity(settings),
        "evaluator_identity": evaluator_identity,
        "levels": {str(power): record for power, record in sorted(levels.items())},
    }


def _evaluator_checkpoint_identity(evaluator: object) -> dict[str, object]:
    """Return a JSON-stable identity for safe checkpoint reuse."""
    identity_method = getattr(evaluator, "checkpoint_identity", None)
    if callable(identity_method):
        identity = identity_method()
        if not isinstance(identity, dict):
            raise TypeError("evaluator.checkpoint_identity() must return a dictionary.")
    else:
        identity = {
            "evaluator_type": f"{type(evaluator).__module__}.{type(evaluator).__qualname__}",
        }
    try:
        serialized = json.dumps(identity, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("The evaluator checkpoint identity must be finite and JSON serializable.") from exc
    return json.loads(serialized)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            temporary_name = handle.name
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _load_checkpoint(
    path: Path,
    stage: Stage,
    settings: FullMomentumIntegrationSettings,
    evaluator_identity: dict[str, object],
) -> dict[int, dict[str, list[float] | list[int]]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = _checkpoint_payload(stage, settings, evaluator_identity, {})
    found_version = payload.get("format_version")
    if found_version in (
        "full_momentum_qsfi_stage_v1",
        "full_momentum_qsfi_stage_v2",
        "full_momentum_qsfi_stage_v3",
        "full_momentum_qsfi_stage_v4",
    ):
        return {}
    if found_version != expected["format_version"]:
        raise ValueError(f"Checkpoint {path} has an incompatible format version.")
    if (
        payload.get("stage") != stage
        or payload.get("integration_identity") != expected["integration_identity"]
    ):
        raise ValueError(f"Checkpoint {path} does not match the requested stage/settings.")
    if payload.get("evaluator_identity") != expected["evaluator_identity"]:
        raise ValueError(f"Checkpoint {path} does not match the requested evaluator identity.")
    raw_levels = payload.get("levels", {})
    if not isinstance(raw_levels, dict):
        raise ValueError(f"Checkpoint {path} has invalid Sobol-level records.")
    levels: dict[int, dict[str, list[float] | list[int]]] = {}
    for raw_power, raw_record in raw_levels.items():
        try:
            power = int(raw_power)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint {path} has an invalid Sobol power.") from exc
        if power < 1 or not isinstance(raw_record, dict):
            raise ValueError(f"Checkpoint {path} has invalid Sobol-level records.")
        values = [float(value) for value in raw_record.get("replicate_integrals", [])]
        excluded = [int(value) for value in raw_record.get("excluded_counts", [])]
        if len(values) != len(excluded):
            raise ValueError(f"Checkpoint {path} has inconsistent replicate records.")
        if any(not math.isfinite(value) for value in values) or any(
            value < 0 for value in excluded
        ):
            raise ValueError(f"Checkpoint {path} contains invalid replicate data.")
        levels[power] = {
            "replicate_integrals": values,
            "excluded_counts": excluded,
        }
    return levels


def _replicate_statistics(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan, math.inf, math.inf
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, math.inf, math.inf
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    if standard_error == 0.0:
        relative_error = 0.0
    elif mean == 0.0:
        relative_error = math.inf
    else:
        relative_error = abs(standard_error / mean)
    return mean, standard_error, relative_error


def _evaluate_point_block(
    points: np.ndarray,
    evaluator: FullMomentumShapeEvaluator,
    stage: Stage,
    settings: FullMomentumIntegrationSettings,
    *,
    replicate: int,
    first_sample_index: int,
) -> tuple[float, int]:
    sample_values = np.empty(points.shape[0], dtype=float)
    excluded = 0
    for local_index, point in enumerate(points):
        try:
            if stage == "bispectrum":
                value = bispectrum_sample_value(point, evaluator, settings)
            else:
                value = trispectrum_sample_value(point, evaluator, settings, stage=stage)
        except RatioDomainError:
            value = 0.0
            excluded += 1
        if not math.isfinite(value):
            sample_index = first_sample_index + local_index
            raise FloatingPointError(
                f"Non-finite {stage} integrand at Sobol replicate {replicate}, "
                f"sample {sample_index}."
            )
        sample_values[local_index] = value
    return float(np.mean(sample_values)), excluded


def _evaluate_replicate(
    evaluator: FullMomentumShapeEvaluator,
    stage: Stage,
    settings: FullMomentumIntegrationSettings,
    *,
    power: int,
    replicate: int,
    previous_integral: float | None,
    previous_excluded: int | None,
) -> tuple[float, int]:
    dimension = 3 if stage == "bispectrum" else 6
    sampler = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=int(settings.sobol_seed + 104729 * replicate),
    )
    points = sampler.random_base2(power)
    if previous_integral is None:
        return _evaluate_point_block(
            points,
            evaluator,
            stage,
            settings,
            replicate=replicate,
            first_sample_index=0,
        )

    prefix_size = 2 ** (power - 1)
    suffix_integral, suffix_excluded = _evaluate_point_block(
        points[prefix_size:],
        evaluator,
        stage,
        settings,
        replicate=replicate,
        first_sample_index=prefix_size,
    )
    integral = 0.5 * (float(previous_integral) + suffix_integral)
    excluded = int(previous_excluded or 0) + suffix_excluded
    return integral, excluded


def estimate_stage(
    evaluator: FullMomentumShapeEvaluator,
    stage: Stage,
    settings: FullMomentumIntegrationSettings,
    *,
    checkpoint_path: Path | str | None = None,
) -> StageEstimate:
    """Estimate one stage with adaptive independent scrambled Sobol replicates.

    Four-point stages add replicates until the requested relative standard
    error is reached. If necessary they increase the nested Sobol level and
    evaluate only the new half of every sequence. A checkpoint is written
    after each replicate and remains reusable when stopping controls change.
    """
    if stage not in ("bispectrum", "contact", "exchange"):
        raise ValueError(f"Unknown projection stage {stage!r}.")
    evaluator_identity = _evaluator_checkpoint_identity(evaluator)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is None:
        levels: dict[int, dict[str, list[float] | list[int]]] = {}
    else:
        levels = _load_checkpoint(
            checkpoint,
            stage,
            settings,
            evaluator_identity,
        )

    initial_power = settings.sobol_power(stage)
    maximum_power = settings.maximum_sobol_power(stage)
    minimum_replicates = settings.minimum_replicates(stage)
    maximum_replicates = settings.maximum_replicates(stage)
    relative_error_target = settings.relative_error_target(stage)
    attempted_powers: list[int] = []
    sampling_history: list[dict[str, object]] = []
    final_power = initial_power
    precision_accepted = False

    for power in range(initial_power, maximum_power + 1):
        attempted_powers.append(power)
        record = levels.setdefault(
            power,
            {"replicate_integrals": [], "excluded_counts": []},
        )
        replicate_integrals = record["replicate_integrals"]
        excluded_counts = record["excluded_counts"]
        assert isinstance(replicate_integrals, list)
        assert isinstance(excluded_counts, list)

        _, _, relative_error = _replicate_statistics(replicate_integrals)
        while len(replicate_integrals) < minimum_replicates or (
            relative_error > relative_error_target
            and len(replicate_integrals) < maximum_replicates
        ):
            replicate = len(replicate_integrals)
            previous_record = levels.get(power - 1)
            previous_integral = None
            previous_excluded = None
            if previous_record is not None:
                previous_values = previous_record["replicate_integrals"]
                previous_counts = previous_record["excluded_counts"]
                assert isinstance(previous_values, list)
                assert isinstance(previous_counts, list)
                if replicate < len(previous_values):
                    previous_integral = float(previous_values[replicate])
                    previous_excluded = int(previous_counts[replicate])
            integral, excluded = _evaluate_replicate(
                evaluator,
                stage,
                settings,
                power=power,
                replicate=replicate,
                previous_integral=previous_integral,
                previous_excluded=previous_excluded,
            )
            replicate_integrals.append(integral)
            excluded_counts.append(excluded)
            _, _, relative_error = _replicate_statistics(replicate_integrals)
            if checkpoint is not None:
                _atomic_write_json(
                    checkpoint,
                    _checkpoint_payload(
                        stage,
                        settings,
                        evaluator_identity,
                        levels,
                    ),
                )
            relative_text = (
                "inf" if not math.isfinite(relative_error) else f"{relative_error:.3%}"
            )
            print(
                f"[adaptive QMC] {stage}: m={power}, replicate "
                f"{len(replicate_integrals)}/{maximum_replicates}, "
                f"relative SE={relative_text}"
            )

        raw_integral, raw_standard_error, relative_error = _replicate_statistics(
            replicate_integrals
        )
        precision_accepted = relative_error <= relative_error_target
        sampling_history.append(
            {
                "sobol_power": power,
                "samples_per_replicate": 2**power,
                "completed_replicates": len(replicate_integrals),
                "relative_standard_error": relative_error,
                "precision_accepted": bool(precision_accepted),
            }
        )
        final_power = power
        if stage == "bispectrum" or precision_accepted:
            break

    final_record = levels[final_power]
    replicate_integrals = final_record["replicate_integrals"]
    excluded_counts = final_record["excluded_counts"]
    assert isinstance(replicate_integrals, list)
    assert isinstance(excluded_counts, list)
    raw_integral, raw_standard_error, relative_error = _replicate_statistics(
        replicate_integrals
    )
    if stage == "bispectrum":
        stop_reason = "fixed_budget"
    elif precision_accepted:
        stop_reason = "target_met"
    else:
        stop_reason = "maximum_power_reached"

    prefactor3, prefactor4, _ = reduced_cumulant_prefactors(settings)
    prefactor = prefactor3 if stage == "bispectrum" else prefactor4
    excluded_total = int(sum(excluded_counts))
    samples_per_replicate = 2**final_power
    sample_total = int(samples_per_replicate * len(replicate_integrals))
    return StageEstimate(
        stage=stage,
        raw_integral=raw_integral,
        raw_standard_error=raw_standard_error,
        coefficient=float(prefactor * raw_integral),
        coefficient_standard_error=float(abs(prefactor) * raw_standard_error),
        samples_per_replicate=samples_per_replicate,
        completed_replicates=len(replicate_integrals),
        sobol_power_used=final_power,
        relative_standard_error=relative_error,
        relative_error_target=relative_error_target,
        precision_accepted=bool(precision_accepted),
        stop_reason=stop_reason,
        attempted_sobol_powers=tuple(attempted_powers),
        sampling_history=tuple(sampling_history),
        excluded_ratio_samples=excluded_total,
        excluded_ratio_fraction=float(excluded_total / sample_total),
        replicate_integrals=tuple(float(value) for value in replicate_integrals),
    )


def project_all_stages(
    evaluator: FullMomentumShapeEvaluator,
    settings: FullMomentumIntegrationSettings,
    *,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run/resume bispectrum, contact, and exchange stages in that order."""
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    estimates: dict[Stage, StageEstimate] = {}
    for stage in ("bispectrum", "contact", "exchange"):
        checkpoint = None if directory is None else directory / f"{stage}.json"
        estimates[stage] = estimate_stage(
            evaluator,
            stage,
            settings,
            checkpoint_path=checkpoint,
        )
    _, _, sigma2 = reduced_cumulant_prefactors(settings)
    sampling_precision_accepted = all(
        estimate.precision_accepted for estimate in estimates.values()
    )
    result = {
        "projection_method": "full_momentum_qmc",
        "shape_equations": {
            "bispectrum": "arXiv:1803.04412 Eq. (20)",
            "contact_trispectrum": "arXiv:1803.04412 Eqs. (30)-(31)",
            "exchange_trispectrum": "arXiv:1803.04412 Eqs. (29)-(32), s+t+u",
        },
        "finite_peak_extension": (
            "sqrt(product of the physical area-normalized lognormal power "
            "on every external leg)"
        ),
        "sigma2": sigma2,
        "sampling_precision_accepted": sampling_precision_accepted,
        "numerical_status": (
            "sampling_target_met" if sampling_precision_accepted else "provisional_sampling_error"
        ),
        "S3_per_lambda3_over_H": estimates["bispectrum"].coefficient,
        "S3_per_lambda3_over_H_standard_error": estimates["bispectrum"].coefficient_standard_error,
        "S4_per_lambda4_contact": estimates["contact"].coefficient,
        "S4_per_lambda4_contact_standard_error": estimates["contact"].coefficient_standard_error,
        "S4_per_lambda3_over_H_squared_exchange": estimates["exchange"].coefficient,
        "S4_per_lambda3_over_H_squared_exchange_standard_error": estimates["exchange"].coefficient_standard_error,
        "stage_estimates": {stage: estimate.as_dict() for stage, estimate in estimates.items()},
        "integration_settings": asdict(settings),
    }
    result.update(settings.sound_speed.as_dict())
    result["sound_speed"] = {
        "ratio": settings.sound_speed.ratio,
        "c_phi": settings.sound_speed.c_phi,
        "c_sigma": settings.sound_speed.c_sigma,
        "i_epsilon": settings.sound_speed.i_epsilon,
    }
    return result


def project_with_mixed_grid(
    grid: object,
    *,
    nutilde: float,
    settings: FullMomentumIntegrationSettings,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, object]:
    """Build a ratio table from ``MixedGrid`` and run all projection stages."""
    nutilde = float(nutilde)
    if not math.isfinite(nutilde) or nutilde <= 0.0:
        raise ValueError("nutilde must be positive and finite.")
    grid_sound_speed = getattr(
        grid,
        "sound_speed",
        SoundSpeedConfiguration(1.0, 1.0, i_epsilon=0.0),
    )
    if grid_sound_speed != settings.sound_speed:
        raise ValueError(
            "MixedGrid and FullMomentumIntegrationSettings use different "
            "sound-speed configurations."
        )
    functions = build_ratio_function_bundle(
        grid,
        minimum_ratio=settings.ratio_floor,
        node_count=settings.ratio_nodes,
    )
    evaluator = FullMomentumShapeEvaluator(
        functions,
        nutilde=nutilde,
        sound_speed=settings.sound_speed,
    )
    record = project_all_stages(
        evaluator,
        settings,
        checkpoint_dir=checkpoint_dir,
    )
    return {
        **record,
        "nutilde": nutilde,
        "ratio_interpolation": {
            "coordinate": "linear interpolation in log(momentum ratio) at fixed base-contour nodes",
            "minimum_ratio": functions.minimum_ratio,
            "maximum_ratio": functions.i_plus.maximum_ratio,
            "node_count": int(functions.i_plus.ratios.size),
        },
    }
