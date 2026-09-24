#!/usr/bin/env python3
"""Complete momentum-dependent QSFI three- and four-point shape kernels.

The time integrals follow arXiv:1803.04412 Eqs. (20) and (29)--(32).
This module contains no PBH abundance or coupling scan logic; it only owns
momentum geometry, ratio interpolation, and correlator shape evaluation.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .sound_speed import SoundSpeedConfiguration

ComplexArray = np.ndarray
CORRELATOR_FORMULA_VERSION = "full_momentum_eq20_30_31_32_sound_speed_v2"


def _hash_array(digest: Any, label: str, values: np.ndarray) -> None:
    """Add an array's label, shape, dtype, and exact bytes to a SHA-256 hash."""
    array = np.ascontiguousarray(values)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(memoryview(array).cast("B"))


class RatioDomainError(ValueError):
    """A required momentum ratio lies outside a precomputed function table."""


def _cumulative_trapezoid(values: ComplexArray, contour: ComplexArray) -> ComplexArray:
    values = np.asarray(values, dtype=np.complex128)
    contour = np.asarray(contour, dtype=np.complex128)
    if values.shape != contour.shape:
        raise ValueError("values and contour must have identical shapes.")
    result = np.zeros_like(values)
    if values.size > 1:
        result[1:] = np.cumsum(
            0.5 * (values[:-1] + values[1:]) * (contour[1:] - contour[:-1])
        )
    return result


def ordered_exchange_terms(
    *,
    z_plus: ComplexArray,
    z_minus: ComplexArray,
    jab_pm: ComplexArray,
    jcd_pp: ComplexArray,
    jcd_pm: ComplexArray,
    jab_pp: ComplexArray,
    jab_mp: ComplexArray,
) -> tuple[complex, complex]:
    r"""Evaluate the ordered and factorized terms of Eq. (32) in O(N)."""
    z_plus = np.asarray(z_plus, dtype=np.complex128)
    z_minus = np.asarray(z_minus, dtype=np.complex128)
    arrays = tuple(np.asarray(value, dtype=np.complex128) for value in (
        jab_pm,
        jcd_pp,
        jcd_pm,
        jab_pp,
        jab_mp,
    ))
    if z_plus.shape != z_minus.shape or any(value.shape != z_plus.shape for value in arrays):
        raise ValueError("All exchange arrays must match the two contour shapes.")
    jab_pm, jcd_pp, jcd_pm, jab_pp, jab_mp = arrays
    ordered_integrand = (
        jab_pm * _cumulative_trapezoid(jcd_pp, z_plus)
        + jcd_pm * _cumulative_trapezoid(jab_pp, z_plus)
    )
    ordered = np.trapz(ordered_integrand, z_plus)
    factorized = np.trapz(jcd_pm, z_plus) * np.trapz(jab_mp, z_minus)
    return complex(ordered), complex(factorized)


@dataclass(frozen=True)
class ComplexRatioTable:
    """Complex arrays tabulated on positive ratios and interpolated in log r."""

    ratios: np.ndarray
    values: np.ndarray
    name: str = "ratio table"

    def __post_init__(self) -> None:
        ratios = np.asarray(self.ratios, dtype=float)
        values = np.asarray(self.values, dtype=np.complex128)
        if ratios.ndim != 1 or ratios.size < 2:
            raise ValueError(f"{self.name} ratios must be a one-dimensional array with at least two nodes.")
        if values.ndim < 2 or values.shape[0] != ratios.size:
            raise ValueError(f"{self.name} values must have one leading row per ratio node.")
        if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0.0) or np.any(np.diff(ratios) <= 0.0):
            raise ValueError(f"{self.name} ratios must be positive, finite, and strictly increasing.")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{self.name} values must be finite.")
        object.__setattr__(self, "ratios", ratios)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_log_ratios", np.log(ratios))

    @property
    def minimum_ratio(self) -> float:
        return float(self.ratios[0])

    @property
    def maximum_ratio(self) -> float:
        return float(self.ratios[-1])

    def evaluate(self, ratio: float) -> ComplexArray:
        """Return the array at ``ratio`` using linear interpolation in log r."""
        ratio = float(ratio)
        tolerance = 32.0 * np.finfo(float).eps
        if not math.isfinite(ratio) or ratio < self.minimum_ratio * (1.0 - tolerance):
            raise RatioDomainError(
                f"{self.name} ratio {ratio!r} is below the table minimum {self.minimum_ratio:g}."
            )
        if ratio > self.maximum_ratio * (1.0 + tolerance):
            raise RatioDomainError(
                f"{self.name} ratio {ratio!r} exceeds the table maximum {self.maximum_ratio:g}."
            )
        ratio = min(max(ratio, self.minimum_ratio), self.maximum_ratio)
        if ratio == self.minimum_ratio:
            return self.values[0].copy()
        if ratio == self.maximum_ratio:
            return self.values[-1].copy()
        log_ratio = math.log(ratio)
        upper = int(np.searchsorted(self._log_ratios, log_ratio, side="right"))
        lower = upper - 1
        fraction = (log_ratio - self._log_ratios[lower]) / (
            self._log_ratios[upper] - self._log_ratios[lower]
        )
        return (1.0 - fraction) * self.values[lower] + fraction * self.values[upper]


@dataclass(frozen=True)
class FourMomentumConfiguration:
    """Four closed three-vectors and their magnitudes."""

    vectors: np.ndarray

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors, dtype=float)
        if vectors.shape != (4, 3) or np.any(~np.isfinite(vectors)):
            raise ValueError("vectors must be a finite array with shape (4, 3).")
        scale = max(1.0, float(np.max(np.linalg.norm(vectors, axis=1))))
        if np.linalg.norm(np.sum(vectors, axis=0)) > 1.0e-11 * scale:
            raise ValueError("The four momentum vectors must close.")
        magnitudes = np.linalg.norm(vectors, axis=1)
        if np.any(magnitudes <= 0.0):
            raise ValueError("All four momentum magnitudes must be positive.")
        immutable = vectors.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "vectors", immutable)
        object.__setattr__(self, "_magnitudes", magnitudes)

    @classmethod
    def from_reduced_coordinates(
        cls,
        k1: float,
        k2: float,
        k3: float,
        *,
        mu12: float,
        mu13: float,
        phi3: float,
    ) -> "FourMomentumConfiguration":
        momenta = np.asarray([k1, k2, k3], dtype=float)
        if np.any(~np.isfinite(momenta)) or np.any(momenta <= 0.0):
            raise ValueError("k1, k2, and k3 must be positive and finite.")
        if not all(math.isfinite(float(value)) for value in (mu12, mu13, phi3)):
            raise ValueError("Angular coordinates must be finite.")
        if not (-1.0 <= mu12 <= 1.0 and -1.0 <= mu13 <= 1.0):
            raise ValueError("mu12 and mu13 must lie in [-1, 1].")
        sin12 = math.sqrt(max(0.0, 1.0 - float(mu12) ** 2))
        sin13 = math.sqrt(max(0.0, 1.0 - float(mu13) ** 2))
        v1 = np.asarray([0.0, 0.0, float(k1)])
        v2 = np.asarray([float(k2) * sin12, 0.0, float(k2) * float(mu12)])
        v3 = np.asarray(
            [
                float(k3) * sin13 * math.cos(float(phi3)),
                float(k3) * sin13 * math.sin(float(phi3)),
                float(k3) * float(mu13),
            ]
        )
        v4 = -(v1 + v2 + v3)
        return cls(np.stack((v1, v2, v3, v4)))

    @property
    def magnitudes(self) -> np.ndarray:
        return np.asarray(self._magnitudes, dtype=float).copy()

    @property
    def total_magnitude(self) -> float:
        return float(np.sum(self._magnitudes))

    @property
    def channel_momenta(self) -> tuple[float, float, float]:
        v = self.vectors
        return (
            float(np.linalg.norm(v[0] + v[1])),
            float(np.linalg.norm(v[0] + v[2])),
            float(np.linalg.norm(v[0] + v[3])),
        )

    def permuted(self, order: Sequence[int]) -> "FourMomentumConfiguration":
        order_tuple = tuple(int(index) for index in order)
        if sorted(order_tuple) != [0, 1, 2, 3]:
            raise ValueError("order must be a permutation of (0, 1, 2, 3).")
        return FourMomentumConfiguration(self.vectors[np.asarray(order_tuple)])

    def scaled(self, factor: float) -> "FourMomentumConfiguration":
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("scale factor must be positive and finite.")
        return FourMomentumConfiguration(factor * self.vectors)


@dataclass(frozen=True)
class RatioFunctionBundle:
    """Mixed propagator and Hankel functions on one base Wick contour."""

    z_plus: np.ndarray
    i_plus: ComplexRatioTable
    h1_plus: ComplexRatioTable
    h2c_plus: ComplexRatioTable

    def __post_init__(self) -> None:
        z_plus = np.asarray(self.z_plus, dtype=np.complex128)
        if z_plus.ndim != 1 or z_plus.size < 3 or np.any(~np.isfinite(z_plus)):
            raise ValueError("z_plus must be a finite one-dimensional contour with at least three points.")
        for table in (self.i_plus, self.h1_plus, self.h2c_plus):
            if table.values.shape[1:] != z_plus.shape:
                raise ValueError(f"{table.name} rows must match the base contour shape.")
            if not np.array_equal(table.ratios, self.i_plus.ratios):
                raise ValueError("All ratio-function tables must use the same ratio nodes.")
        immutable = z_plus.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "z_plus", immutable)

    @property
    def z_minus(self) -> np.ndarray:
        return np.conj(self.z_plus)

    @property
    def minimum_ratio(self) -> float:
        return self.i_plus.minimum_ratio

    def I(self, branch: str, ratio: float) -> ComplexArray:
        values = self.i_plus.evaluate(ratio)
        if branch == "+":
            return values
        if branch == "-":
            return np.conj(values)
        raise ValueError("branch must be '+' or '-'.")

    def H1(self, branch: str, ratio: float) -> ComplexArray:
        if branch == "+":
            return self.h1_plus.evaluate(ratio)
        if branch == "-":
            return np.conj(self.h2c_plus.evaluate(ratio))
        raise ValueError("branch must be '+' or '-'.")

    def H2c(self, branch: str, ratio: float) -> ComplexArray:
        if branch == "+":
            return self.h2c_plus.evaluate(ratio)
        if branch == "-":
            return np.conj(self.h1_plus.evaluate(ratio))
        raise ValueError("branch must be '+' or '-'.")


def build_ratio_function_bundle(
    grid: object,
    *,
    minimum_ratio: float,
    node_count: int,
) -> RatioFunctionBundle:
    """Sample an existing ``MixedGrid`` on logarithmic momentum-ratio nodes."""
    minimum_ratio = float(minimum_ratio)
    node_count = int(node_count)
    if not math.isfinite(minimum_ratio) or not (0.0 < minimum_ratio < 1.0):
        raise ValueError("minimum_ratio must satisfy 0 < minimum_ratio < 1.")
    if node_count < 3:
        raise ValueError("node_count must be at least three.")
    if not hasattr(grid, "z_plus") or not hasattr(grid, "propagator"):
        raise TypeError("grid must expose z_plus and propagator(ratio).")
    ratios = np.geomspace(minimum_ratio, 1.0, node_count)
    i_rows: list[np.ndarray] = []
    h1_rows: list[np.ndarray] = []
    h2_rows: list[np.ndarray] = []
    for ratio in ratios:
        values = grid.propagator(float(ratio))  # type: ignore[attr-defined]
        i_rows.append(np.asarray(values.Iplus, dtype=np.complex128))
        h1_rows.append(np.asarray(values.H1_plus, dtype=np.complex128))
        h2_rows.append(np.asarray(values.H2c_plus, dtype=np.complex128))
    return RatioFunctionBundle(
        z_plus=np.asarray(grid.z_plus, dtype=np.complex128),  # type: ignore[attr-defined]
        i_plus=ComplexRatioTable(ratios, np.asarray(i_rows), name="I_+(r z)"),
        h1_plus=ComplexRatioTable(ratios, np.asarray(h1_rows), name="H_nu^(1)(r z)"),
        h2c_plus=ComplexRatioTable(ratios, np.asarray(h2_rows), name="H_(nu*)^(2)(r z)"),
    )


@dataclass(frozen=True)
class FullMomentumShapeEvaluator:
    """Complete scale-invariant QSFI shape kernels with couplings removed."""

    functions: RatioFunctionBundle
    nutilde: float
    sound_speed: SoundSpeedConfiguration = SoundSpeedConfiguration(
        1.0,
        1.0,
        i_epsilon=0.0,
    )

    def __post_init__(self) -> None:
        nutilde = float(self.nutilde)
        if not math.isfinite(nutilde) or nutilde <= 0.0:
            raise ValueError("nutilde must be positive and finite.")
        if not isinstance(self.sound_speed, SoundSpeedConfiguration):
            raise TypeError("sound_speed must be a SoundSpeedConfiguration.")
        object.__setattr__(self, "nutilde", nutilde)

    def checkpoint_identity(self) -> dict[str, object]:
        """Fingerprint all scientific inputs that determine a shape value."""
        digest = hashlib.sha256()
        _hash_array(digest, "z_plus", self.functions.z_plus)
        _hash_array(digest, "ratios", self.functions.i_plus.ratios)
        _hash_array(digest, "i_plus", self.functions.i_plus.values)
        _hash_array(digest, "h1_plus", self.functions.h1_plus.values)
        _hash_array(digest, "h2c_plus", self.functions.h2c_plus.values)
        identity = {
            "evaluator_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "correlator_formula_version": CORRELATOR_FORMULA_VERSION,
            "nutilde": self.nutilde,
            "ratio_floor": self.functions.minimum_ratio,
            "ratio_nodes": int(self.functions.i_plus.ratios.size),
            "scientific_arrays_sha256": digest.hexdigest(),
        }
        identity.update(self.sound_speed.as_dict())
        return identity

    @staticmethod
    def _positive_momenta(values: Sequence[float], expected: int) -> np.ndarray:
        momenta = np.asarray(values, dtype=float)
        if momenta.shape != (expected,) or np.any(~np.isfinite(momenta)) or np.any(momenta <= 0.0):
            raise ValueError(f"Expected {expected} positive finite momentum magnitudes.")
        return momenta

    def bispectrum_shape(self, k1: float, k2: float, k3: float) -> float:
        r"""Return the power-stripped full Eq. (20) kernel per ``l2^3 l3``.

        Eqs. (18) and (20) give

        ``b = -pi^2/512 * q^2/(ka kb) * Im int dz/z^4 prod I``.

        Choosing ``q=max(k_i)`` keeps all tabulated propagator ratios at most
        one and does not alter the scale-invariant result.
        """
        momenta = np.sort(self._positive_momenta((k1, k2, k3), 3))
        ka, kb, q = (float(value) for value in momenta)
        ratios = (ka / q, kb / q, 1.0)
        z = self.functions.z_plus
        integrand = (
            self.functions.I("+", ratios[0])
            * self.functions.I("+", ratios[1])
            * self.functions.I("+", ratios[2])
            / z**4
        )
        time_integral = complex(np.trapz(integrand, z))
        sound_speed_prefactor = (
            self.sound_speed.ratio**3
            * self.sound_speed.c_phi ** (-1.5)
        )
        result = (
            -(math.pi**2 / 512.0)
            * sound_speed_prefactor
            * (q**2 / (ka * kb))
            * time_integral.imag
        )
        if not math.isfinite(result):
            raise FloatingPointError("The full bispectrum shape is not finite.")
        return float(result)

    def contact_kernel(self, k1: float, k2: float, k3: float, k4: float) -> float:
        r"""Return the complete contact time kernel ``t_c`` of Eq. (31)."""
        momenta = self._positive_momenta((k1, k2, k3, k4), 4)
        total = float(np.sum(momenta))
        ratios = momenta / total
        z = self.functions.z_plus
        product = np.ones_like(z, dtype=np.complex128)
        for ratio in ratios:
            product *= self.functions.I("+", float(ratio))
        time_integral = complex(np.trapz(product / z**4, z))
        result = time_integral.imag
        if not math.isfinite(result):
            raise FloatingPointError("The full contact trispectrum kernel is not finite.")
        return float(result)

    def contact_shape_per_lambda4(self, k1: float, k2: float, k3: float, k4: float) -> float:
        r"""Return the power-stripped Eq. (30) kernel per ``l2^4 lambda4``."""
        # pi^3/2^15 multiplied by the contact coefficient 4/pi.
        sound_speed_prefactor = self.sound_speed.ratio**5 * self.sound_speed.c_phi**3
        return float(
            (math.pi**2 / 8192.0)
            * sound_speed_prefactor
            * self.contact_kernel(k1, k2, k3, k4)
        )

    def exchange_channel_kernel(
        self,
        configuration: FourMomentumConfiguration,
        first_pair: tuple[int, int],
        second_pair: tuple[int, int],
    ) -> float:
        r"""Return one K-scaled exchange channel of Eq. (32).

        With ``y=(k_channel/K) z``, the two time Jacobians cancel the
        explicit ``(k_channel/K)^3`` against the two ``J`` rescalings. This
        equivalent form keeps every mixed-propagator argument at ``k_i/K``.
        """
        indices = tuple(first_pair) + tuple(second_pair)
        if sorted(indices) != [0, 1, 2, 3] or len(set(first_pair)) != 2 or len(set(second_pair)) != 2:
            raise ValueError("The two channel pairs must partition the four external legs.")
        momenta = configuration.magnitudes
        total = configuration.total_magnitude
        external_ratios = momenta / total
        internal_vector = configuration.vectors[first_pair[0]] + configuration.vectors[first_pair[1]]
        internal_ratio = float(np.linalg.norm(internal_vector) / total)
        if internal_ratio < self.functions.minimum_ratio:
            raise RatioDomainError(
                f"Exchange ratio k_channel/K={internal_ratio:g} is below the table floor "
                f"{self.functions.minimum_ratio:g}."
            )

        z_plus = self.functions.z_plus
        z_minus = self.functions.z_minus
        h1_plus = self.functions.H1("+", internal_ratio)
        h2c_plus = self.functions.H2c("+", internal_ratio)
        h1_minus = self.functions.H1("-", internal_ratio)

        def external_product(pair: tuple[int, int], branch: str) -> ComplexArray:
            return (
                self.functions.I(branch, float(external_ratios[pair[0]]))
                * self.functions.I(branch, float(external_ratios[pair[1]]))
            )

        first_plus = external_product(first_pair, "+")
        second_plus = external_product(second_pair, "+")
        first_minus = external_product(first_pair, "-")
        common_plus = z_plus ** (-2.5)
        jab_pm = common_plus * first_plus * h2c_plus
        jab_pp = common_plus * first_plus * h1_plus
        jcd_pm = common_plus * second_plus * h2c_plus
        jcd_pp = common_plus * second_plus * h1_plus
        jab_mp = z_minus ** (-2.5) * first_minus * h1_minus

        ordered, factorized = ordered_exchange_terms(
            z_plus=z_plus,
            z_minus=z_minus,
            jab_pm=jab_pm,
            jcd_pp=jcd_pp,
            jcd_pm=jcd_pm,
            jab_pp=jab_pp,
            jab_mp=jab_mp,
        )
        result = math.exp(-math.pi * self.nutilde) * float(np.real(ordered - factorized))
        if not math.isfinite(result):
            raise FloatingPointError("The full exchange-channel kernel is not finite.")
        return result

    def exchange_kernel(self, configuration: FourMomentumConfiguration) -> float:
        """Return ``t_s+t_t+t_u`` with the three complete exchange channels."""
        pairings = (
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        )
        result = sum(
            self.exchange_channel_kernel(configuration, first_pair, second_pair)
            for first_pair, second_pair in pairings
        )
        if not math.isfinite(result):
            raise FloatingPointError("The summed full exchange kernel is not finite.")
        return float(result)

    def exchange_shape_per_lambda3_squared(self, configuration: FourMomentumConfiguration) -> float:
        r"""Return the power-stripped Eq. (30) kernel per ``l2^4 (lambda3/H)^2``."""
        sound_speed_prefactor = self.sound_speed.ratio**5 * self.sound_speed.c_phi**3
        return float(
            (math.pi**3 / 2.0**15)
            * sound_speed_prefactor
            * self.exchange_kernel(configuration)
        )
