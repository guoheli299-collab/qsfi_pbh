"""Sound-speed configuration and the general QSFI mixing coefficient.

The definitions follow arXiv:1803.04412 Eqs. (11)--(13), with
``r = c_phi / c_sigma``.  The three public benchmark configurations are
``r = 0.3, 1, 3`` and keep both individual sound speeds subluminal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mpmath as mp


def _number_tag(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class SoundSpeedConfiguration:
    """Validated sound speeds and the lower-half-plane branch prescription."""

    c_phi: float
    c_sigma: float
    i_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        c_phi = float(self.c_phi)
        c_sigma = float(self.c_sigma)
        i_epsilon = float(self.i_epsilon)
        if not math.isfinite(c_phi) or not 0.0 < c_phi <= 1.0:
            raise ValueError("c_phi must be finite and satisfy 0 < c_phi <= 1.")
        if not math.isfinite(c_sigma) or not 0.0 < c_sigma <= 1.0:
            raise ValueError("c_sigma must be finite and satisfy 0 < c_sigma <= 1.")
        if not math.isfinite(i_epsilon) or not 0.0 <= i_epsilon <= 1.0e-3:
            raise ValueError("i_epsilon must be finite and satisfy 0 <= i_epsilon <= 1e-3.")
        object.__setattr__(self, "c_phi", c_phi)
        object.__setattr__(self, "c_sigma", c_sigma)
        object.__setattr__(self, "i_epsilon", i_epsilon)

    @property
    def ratio(self) -> float:
        """Return ``r = c_phi / c_sigma``."""

        return self.c_phi / self.c_sigma

    @property
    def complex_ratio(self) -> complex:
        """Return the regulated ratio used across the hypergeometric cut."""

        return complex(self.ratio, -self.i_epsilon)

    @property
    def cache_tag(self) -> str:
        """Stable, filesystem-safe identity for sound-speed-dependent data."""

        return (
            f"r{_number_tag(self.ratio)}_"
            f"cphi{_number_tag(self.c_phi)}_"
            f"csigma{_number_tag(self.c_sigma)}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "sound_speed_ratio": self.ratio,
            "c_phi": self.c_phi,
            "c_sigma": self.c_sigma,
            "sound_speed_i_epsilon": self.i_epsilon,
        }


SOUND_SPEED_CASES = (
    SoundSpeedConfiguration(c_phi=0.3, c_sigma=1.0),
    SoundSpeedConfiguration(c_phi=1.0, c_sigma=1.0),
    SoundSpeedConfiguration(c_phi=1.0, c_sigma=1.0 / 3.0),
)


def generalized_cnu(
    nu: complex,
    sound_speed_ratio: float,
    *,
    i_epsilon: float = 1.0e-8,
    dps: int = 50,
) -> complex:
    r"""Return the general mixing constant :math:`C_\nu^r`.

    This is Eq. (13) of arXiv:1803.04412.  A small negative imaginary part
    in ``r`` selects the branch continuously used by the mixed propagator.
    The unit-ratio, zero-regulator identity is evaluated in closed form so
    that the legacy ``r=1`` calculation remains an exact regression path.
    """

    ratio = float(sound_speed_ratio)
    epsilon = float(i_epsilon)
    precision = int(dps)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("sound_speed_ratio must be finite and positive.")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0e-3:
        raise ValueError("i_epsilon must be finite and satisfy 0 <= i_epsilon <= 1e-3.")
    if precision < 15:
        raise ValueError("dps must be at least 15.")

    with mp.workdps(precision):
        order = mp.mpc(nu)
        if ratio == 1.0 and epsilon == 0.0:
            return complex(mp.sqrt(2 * mp.pi) / mp.cos(mp.pi * order))

        regulated_ratio = mp.mpc(ratio, -epsilon)

        def branch(branch_order: mp.mpc) -> mp.mpc:
            return (
                mp.power(2, branch_order)
                * mp.power(regulated_ratio, branch_order - mp.mpf("0.5"))
                / mp.pi
                * mp.gamma(mp.mpf("0.5") - branch_order)
                * mp.gamma(branch_order)
                * mp.hyp2f1(
                    mp.mpf("0.25") - branch_order / 2,
                    mp.mpf("0.75") - branch_order / 2,
                    1 - branch_order,
                    1 / regulated_ratio**2,
                )
            )

        return complex(branch(order) + branch(-order))
