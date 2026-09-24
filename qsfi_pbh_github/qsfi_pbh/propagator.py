#!/usr/bin/env python3
r"""Direct general-sound-speed QSFI clock coefficients and PBH projections.

The squeezed, triangular and collapsed coefficients follow arXiv:1803.04412
Eqs. (22), (34)--(36), and (40).  Every coefficient is evaluated directly on
the half-way Wick-rotated contour described in arXiv:1703.10166.  The mixed
propagator uses Eq. (11) in a reverse-tail form, which is equivalent to the
printed Eq. (12), including the complex conjugate on ``B^{r*}_{nu,mp}``, but
does not suffer catastrophic UV cancellation.

For the heavy clock, ``nu=i*nu_tilde``.  Plus vertices use the lower contour;
minus vertices use its upper-half-plane conjugate.  The triangular exchange
keeps the original Schwinger-Keldysh branch assignments and O(N) cumulative
quadrature.  No published-vector-data or Abel-extrapolation backend is used.

Figures use the printed sine--cosine conventions of Ref. 1803 Eqs. (37) and
(41).  Use ``--convergence`` to compare deeper/finer shifted-Wick contours.

Dependencies:
  pip install numpy mpmath matplotlib

Publication output:
  The default figure has no equation-number labels on the panels.
  It saves both PNG and PDF for paper use.
  The default plot uses the physical prefactors and is not panel-normalized.

Default article-plot physical parameters:
  peak_height P_zeta(k_*) = 1e-2
  lognormal_delta = 0.5
  equivalent power_area = sqrt(2*pi)*Delta*P_zeta(k_*)
  lambda2_over_H = 0.38
  lambda3_over_H = 1.0
  lambda4 = pi/4


Examples:
  python qsfi_article_4x3_paper_figure_physical.py --mode all --convergence
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import mpmath as mp

from .collapse import QNG_MAX, qng_from_cumulants
from .config import ProductionConfig

from .cumulants import (
    FullMomentumIntegrationSettings,
    project_with_mixed_grid,
)
from .momentum_shapes import CORRELATOR_FORMULA_VERSION


_PRODUCTION_CONFIG = ProductionConfig()
from .spectrum import (
    area_from_peak_height,
    density_variance_response,
    peak_height_from_area,
)
from .sound_speed import SoundSpeedConfiguration, generalized_cnu
ComplexArray = np.ndarray
PM = Literal["+", "-"]
CACHE_VERSION = "v8_sound_speed_shifted_wick_eq11_tail"
CLOCK_FORMULA_VERSION = "general_r_shifted_wick_eq11_22_34_35_36_40_v8_sincos"
FORMULA_VERSION = f"{CORRELATOR_FORMULA_VERSION}_qmc_v3_physical_lognormal_legs"


def sound_speed_from_args(args: argparse.Namespace | object) -> SoundSpeedConfiguration:
    """Build the validated sound-speed configuration stored in a namespace."""

    return SoundSpeedConfiguration(
        c_phi=float(getattr(args, "c_phi", 1.0)),
        c_sigma=float(getattr(args, "c_sigma", 1.0)),
        i_epsilon=float(getattr(args, "sound_speed_i_epsilon", 0.0)),
    )


def full_momentum_contour_tag(args: argparse.Namespace) -> str:
    """Stable cache identity for the mixed-propagator contour backend."""
    sound_speed = sound_speed_from_args(args)
    return (
        f"{sound_speed.cache_tag}_eps{sound_speed.i_epsilon:g}_"
        f"p{args.wick_pivot:g}_y{args.wick_depth:g}_tail{args.tail_depth:g}_"
        f"dx{args.real_step:g}_dy{args.vertical_step:g}_nl{args.nlog}_dps{args.dps}"
    )


def full_momentum_checkpoint_tag(args: argparse.Namespace) -> str:
    """Stable QMC checkpoint identity for sample-defining settings."""

    return (
        f"{full_momentum_contour_tag(args)}_"
        f"delta{float(args.lognormal_delta):.12g}_"
        f"kpeakR{float(args.k_peak_r):.12g}_"
        f"sigmarange{float(args.full_momentum_sigma_range):.12g}_"
        f"area{float(args.power_area):.12g}_"
        f"lambda2{float(args.lambda2_over_H):.12g}_"
        f"ratiofloor{float(args.full_ratio_floor):.12g}_"
        f"rationodes{int(args.full_ratio_nodes)}_"
        f"radial{str(args.radial_sampling)}_"
        f"seed{int(args.sobol_seed)}"
    )


def cumtrapz_complex(f: ComplexArray, z: ComplexArray) -> ComplexArray:
    """Cumulative contour integral using the trapezoid rule."""
    out = np.zeros_like(f, dtype=np.complex128)
    if len(f) <= 1:
        return out
    dz = z[1:] - z[:-1]
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * dz)
    return out


def reverse_cumtrapz_complex(f: ComplexArray, z: ComplexArray) -> ComplexArray:
    """Return contour tails ``out[i] = integral(z[i], z[-1]) f dz``.

    The reverse accumulation is deliberate: forming ``total-prefix`` loses all
    useful digits when the desired UV tail is much smaller than the two terms.
    """
    f = np.asarray(f, dtype=np.complex128)
    z = np.asarray(z, dtype=np.complex128)
    if f.shape != z.shape:
        raise ValueError("f and z must have the same shape.")
    out = np.zeros_like(f, dtype=np.complex128)
    if f.size <= 1:
        return out
    increments = 0.5 * (f[:-1] + f[1:]) * (z[1:] - z[:-1])
    out[:-1] = np.cumsum(increments[::-1])[::-1]
    return out


def build_shifted_wick_contour(
    *,
    xmin: float,
    pivot: float,
    depth: float,
    nlog: int,
    real_step: float,
    vertical_step: float,
) -> tuple[ComplexArray, ComplexArray, int]:
    """Build the half-way Wick contours of Ref. 1703.10166 Fig. 1.

    The plus branch follows the positive real axis to ``pivot`` and then runs
    vertically into the fourth quadrant.  The minus branch is its conjugate.
    ``real_count`` includes the shared pivot point.
    """
    xmin = float(xmin)
    pivot = float(pivot)
    depth = float(depth)
    real_step = float(real_step)
    vertical_step = float(vertical_step)
    nlog = int(nlog)
    if not (0.0 < xmin < pivot) or depth <= 0.0:
        raise ValueError("Require 0 < xmin < pivot and depth > 0.")
    if nlog < 3 or real_step <= 0.0 or vertical_step <= 0.0:
        raise ValueError("Require nlog >= 3 and positive contour step sizes.")

    log_end = min(1.0, pivot)
    logarithmic = np.geomspace(xmin, log_end, nlog)
    if math.isclose(log_end, pivot, rel_tol=0.0, abs_tol=1.0e-15):
        real = logarithmic
    else:
        nreal_linear = int(math.ceil((pivot - log_end) / real_step)) + 1
        linear = np.linspace(log_end, pivot, nreal_linear)
        real = np.concatenate((logarithmic[:-1], linear))

    nvertical = int(math.ceil(depth / vertical_step)) + 1
    vertical_y = np.linspace(0.0, depth, nvertical)
    plus = np.concatenate((real[:-1].astype(np.complex128), pivot - 1j * vertical_y))
    return plus, np.conj(plus), int(real.size)


def cpow(z: ComplexArray | complex | float, alpha: complex | float) -> ComplexArray | complex:
    """Principal-branch complex power."""
    return np.exp(alpha * np.log(z))


def mixed_propagator_constants(nu: complex, cnu: complex) -> tuple[complex, complex]:
    """Return the two constant brackets in ``I_+``/``I_-``.

    For a heavy field, ``nu=i*nu_tilde``, the coefficient multiplying
    ``H^(1)_nu`` must be the complex conjugate of the coefficient multiplying
    ``H^(2)_{nu*}``.  This is fixed independently by the small-z expansion in
    Ref. 1803 Eq. (15).  Conjugating only ``C_nu`` in the printed shorthand of
    Eq. (12) misses the conjugation of the phase when ``nu`` is imaginary and
    fails that expansion by an order-one complex term.
    """
    h2_constant = complex(np.exp(+1j * np.pi * (1 - 2 * nu) / 4.0) * cnu)
    return complex(np.conj(h2_constant)), h2_constant


def _json_scalar(value):
    """Convert numpy scalars to plain JSON-friendly values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def cache_metadata_matches(found: Dict[str, object], expected: Dict[str, object]) -> bool:
    """Return True when all expected cache metadata fields match."""
    for key, exp in expected.items():
        if key not in found:
            return False
        got = _json_scalar(found[key])
        exp = _json_scalar(exp)
        if isinstance(exp, float):
            try:
                if not math.isclose(float(got), exp, rel_tol=0.0, abs_tol=1e-15):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if got != exp:
                return False
    return True


def normalized_curve_stats(y_ref: np.ndarray, y_new: np.ndarray) -> Dict[str, float]:
    """Compare two curves after max-absolute normalization."""
    def normalize(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        m = np.nanmax(np.abs(y))
        return y / m if np.isfinite(m) and m > 0 else y

    a = normalize(y_ref)
    b = normalize(y_new)
    if a.shape != b.shape:
        raise ValueError(f"curve shapes differ: {a.shape} vs {b.shape}")
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "max_abs_diff": float(np.nanmax(np.abs(a - b))),
        "rms_diff": float(np.sqrt(np.nanmean((a - b) ** 2))),
        "corr": corr,
    }


def run_metadata(args: argparse.Namespace | object | None = None) -> Dict[str, object]:
    """Metadata that makes coefficient JSON files self-describing."""
    meta: Dict[str, object] = {
        "formula_version": CLOCK_FORMULA_VERSION,
        "plot_convention_fig5": "printed Ref.1803 Eq.(37): t1*sin+t2*cos",
        "plot_convention_fig8": "printed Ref.1803 Eq.(41): c1*sin+c2*cos",
        "plot_uses_fig5": "t1/t2 printed sine-cosine form",
        "plot_uses_fig8": "c1/c2 printed sine-cosine form",
        "coefficient_method": "direct half-way shifted-Wick contour integration",
    }
    if args is not None:
        for key in (
            "wick_pivot",
            "wick_depth",
            "tail_depth",
            "real_step",
            "vertical_step",
            "nlog",
            "dps",
            "xmin",
        ):
            if hasattr(args, key):
                meta[key] = _json_scalar(getattr(args, key))
        meta.update(sound_speed_from_args(args).as_dict())
    return meta


def attach_record_metadata(mode: str, results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    """Annotate coefficient records with the plotting convention used."""
    if mode == "fig5":
        note = "plotted curve uses the printed Ref.1803 Eq.(37) convention: t1*sin+t2*cos."
        plot_uses = "t1/t2 printed sine-cosine form"
    elif mode == "fig8":
        note = "plotted curve uses the printed Ref.1803 Eq.(41) convention: c1*sin+c2*cos."
        plot_uses = "c1/c2 printed sine-cosine form"
    else:
        note = "s1/s2 printed sin-cos form is the plotted convention for Fig.2."
        plot_uses = "s1/s2 printed sin-cos form"

    annotated: Dict[str, Dict[str, object]] = {}
    for key, record in results.items():
        annotated[key] = dict(record)
        annotated[key]["plot_uses"] = plot_uses
        annotated[key]["diagnostic_note"] = note
    return annotated


def write_coefficients_json(
    outdir: Path,
    new_results: Dict[str, Dict[str, Dict[str, float]]],
    args: argparse.Namespace | object | None = None,
) -> None:
    """Write per-mode coefficient files and merge the combined coefficient JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    combined_path = outdir / "coefficients_r1.json"
    if combined_path.exists():
        try:
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
            if not isinstance(combined, dict):
                combined = {}
        except json.JSONDecodeError:
            combined = {}
    else:
        combined = {}

    combined["metadata"] = run_metadata(args)
    for mode, results in new_results.items():
        annotated = attach_record_metadata(mode, results)
        combined[mode] = annotated
        with open(outdir / f"coefficients_r1_{mode}.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": run_metadata(args), mode: annotated}, f, indent=2, ensure_ascii=False, sort_keys=True)

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, sort_keys=True)


def should_exclude_from_zip(path: Path) -> bool:
    parts = set(path.parts)
    return path.name == ".DS_Store" or "__MACOSX" in parts or ".mplconfig" in parts


def write_clean_zip(outdir: Path, zip_path: Path | None = None) -> Path:
    """Create a clean zip of output artifacts, excluding local OS/cache files."""
    outdir = outdir.resolve()
    if zip_path is None:
        zip_path = outdir.with_name(f"{outdir.name}_clean.zip")
    else:
        zip_path = zip_path.resolve()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(outdir.rglob("*")):
            if path.is_dir() or should_exclude_from_zip(path.relative_to(outdir)):
                continue
            arcname = Path(outdir.name) / path.relative_to(outdir)
            zf.write(path, arcname.as_posix())
    return zip_path


def hankel1_vec(nu: complex, z: ComplexArray, dps: int) -> ComplexArray:
    mp.mp.dps = dps
    nu_mp = mp.mpc(nu)
    return np.asarray([complex(mp.hankel1(nu_mp, mp.mpc(zz))) for zz in z], dtype=np.complex128)


def hankel2_order_conj_vec(nu: complex, z: ComplexArray, dps: int) -> ComplexArray:
    """H^{(2)}_{nu*}(z), matching the notation in Eq. (12)."""
    mp.mp.dps = dps
    nuc_mp = mp.conj(mp.mpc(nu))
    return np.asarray([complex(mp.hankel2(nuc_mp, mp.mpc(zz))) for zz in z], dtype=np.complex128)


@dataclass(frozen=True)
class MixedPropagatorValues:
    """Mixed-propagator and Hankel arrays on conjugate argument contours."""

    ratio: float
    z_plus: ComplexArray
    z_minus: ComplexArray
    H1_plus: ComplexArray
    H2c_plus: ComplexArray
    H1_minus: ComplexArray
    H2c_minus: ComplexArray
    Iplus: ComplexArray
    Iminus: ComplexArray


@dataclass
class MixedGrid:
    """Half-way Wick contour and stable Ref. 1803 Eq. (11) propagators."""

    nutilde: float
    sound_speed: SoundSpeedConfiguration = SoundSpeedConfiguration(
        1.0,
        1.0,
        i_epsilon=0.0,
    )
    pivot: float = 4.0
    depth: float = 14.0
    real_step: float = 0.025
    vertical_step: float = 0.025
    xmin: float = 1e-6
    nlog: int = 500
    tail_depth: float = 16.0
    dps: int = 50
    cache_dir: Path | None = Path("cache")
    force: bool = False

    def __post_init__(self) -> None:
        self.nutilde = float(self.nutilde)
        if not isinstance(self.sound_speed, SoundSpeedConfiguration):
            raise TypeError("sound_speed must be a SoundSpeedConfiguration.")
        self.nu: complex = 1j * self.nutilde
        self.Cnu: complex = generalized_cnu(
            self.nu,
            self.sound_speed.ratio,
            i_epsilon=self.sound_speed.i_epsilon,
            dps=self.dps,
        )
        self.z_plus, self.z_minus, self.real_count = build_shifted_wick_contour(
            xmin=self.xmin,
            pivot=self.pivot,
            depth=self.depth,
            nlog=self.nlog,
            real_step=self.real_step,
            vertical_step=self.vertical_step,
        )
        self._propagators: Dict[float, MixedPropagatorValues] = {}
        base = self.propagator(1.0)

        # Public aliases used by the coefficient and plotting routines.
        self.H1_plus = base.H1_plus
        self.H2c_plus = base.H2c_plus
        self.H1_minus = base.H1_minus
        self.H2c_minus = base.H2c_minus
        self.Iplus = base.Iplus
        self.Iminus = base.Iminus

    def cache_metadata(self, momentum_ratio: float = 1.0) -> Dict[str, object]:
        metadata = {
            "cache_version": CACHE_VERSION,
            "formula_version": CLOCK_FORMULA_VERSION,
            "contour": "half_way_shifted_wick",
            "nutilde": float(self.nutilde),
            "momentum_ratio": float(momentum_ratio),
            "pivot": float(self.pivot),
            "depth": float(self.depth),
            "tail_depth": float(self.tail_depth),
            "real_step": float(self.real_step),
            "vertical_step": float(self.vertical_step),
            "xmin": float(self.xmin),
            "nlog": int(self.nlog),
            "dps": int(self.dps),
            "actual_grid_points": int(self.z_plus.size),
        }
        metadata.update(self.sound_speed.as_dict())
        return metadata

    def cache_path(self, momentum_ratio: float = 1.0) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = (
            f"{CACHE_VERSION}_{self.sound_speed.cache_tag}_nu{self.nutilde:g}_"
            f"q{momentum_ratio:g}_p{self.pivot:g}_"
            f"y{self.depth:g}_tail{self.tail_depth:g}_dx{self.real_step:g}_"
            f"dy{self.vertical_step:g}_nl{self.nlog}_xmin{self.xmin:g}_dps{self.dps}.npz"
        ).replace("+", "p").replace("-", "m")
        return self.cache_dir / tag

    def _extended_argument_contour(self, momentum_ratio: float) -> tuple[ComplexArray, int]:
        z_main = momentum_ratio * self.z_plus
        current_depth = float(-z_main[-1].imag)
        target_depth = max(float(self.tail_depth), current_depth)
        if target_depth <= current_depth + 1.0e-14:
            return z_main, z_main.size

        extension_step = min(float(self.vertical_step), 0.05)
        next_depth = current_depth + extension_step
        if next_depth >= target_depth:
            extra_depths = np.asarray([target_depth])
        else:
            count = int(math.ceil((target_depth - next_depth) / extension_step)) + 1
            extra_depths = np.linspace(next_depth, target_depth, count)
        real_part = momentum_ratio * float(self.pivot)
        extension = real_part - 1j * extra_depths
        return np.concatenate((z_main, extension)), z_main.size

    def _build_propagator(self, momentum_ratio: float) -> MixedPropagatorValues:
        z_work, main_size = self._extended_argument_contour(momentum_ratio)
        H1_work = hankel1_vec(self.nu, z_work, self.dps)
        H2c_work = hankel2_order_conj_vec(self.nu, z_work, self.dps)
        invsqrtz = 1.0 / np.sqrt(z_work)

        # Eq. (11), plus branch.  The H1 coefficient is imposed as its
        # exponentially damped UV tail.  This is algebraically equivalent to
        # the printed B^{r*}_{nu,-} bracket in Eq. (12) on the real segment,
        # without subtracting two exponentially large nearly equal numbers.
        phase = np.exp(-1j * self.sound_speed.complex_ratio * z_work)
        f2 = invsqrtz * H2c_work * phase
        f1 = invsqrtz * H1_work * phase
        h1_bracket = -1j * reverse_cumtrapz_complex(f2, z_work)
        _, h2_constant = mixed_propagator_constants(self.nu, self.Cnu)
        h2_bracket = h2_constant - 1j * cumtrapz_complex(f1, z_work)
        pref = np.exp(-np.pi * self.nutilde) * cpow(z_work, 1.5)
        Iplus_work = pref * (H1_work * h1_bracket + H2c_work * h2_bracket)

        z_plus = z_work[:main_size]
        H1_plus = H1_work[:main_size]
        H2c_plus = H2c_work[:main_size]
        Iplus = Iplus_work[:main_size]
        return MixedPropagatorValues(
            ratio=float(momentum_ratio),
            z_plus=z_plus,
            z_minus=np.conj(z_plus),
            H1_plus=H1_plus,
            H2c_plus=H2c_plus,
            H1_minus=np.conj(H2c_plus),
            H2c_minus=np.conj(H1_plus),
            Iplus=Iplus,
            Iminus=np.conj(Iplus),
        )

    def propagator(self, momentum_ratio: float = 1.0) -> MixedPropagatorValues:
        momentum_ratio = float(momentum_ratio)
        if not math.isfinite(momentum_ratio) or momentum_ratio <= 0.0:
            raise ValueError("The mixed-propagator momentum ratio must be positive and finite.")
        key = float(momentum_ratio)
        if key in self._propagators:
            return self._propagators[key]

        path = self.cache_path(key)
        values: MixedPropagatorValues | None = None
        if path is not None and path.exists() and not self.force:
            with np.load(path) as data:
                try:
                    found_meta = json.loads(str(data["metadata"].item()))
                except (KeyError, json.JSONDecodeError, ValueError, AttributeError):
                    found_meta = {}
                if cache_metadata_matches(found_meta, self.cache_metadata(key)):
                    z_plus = key * self.z_plus
                    H1_plus = data["H1_plus"]
                    H2c_plus = data["H2c_plus"]
                    Iplus = data["Iplus"]
                    values = MixedPropagatorValues(
                        ratio=key,
                        z_plus=z_plus,
                        z_minus=np.conj(z_plus),
                        H1_plus=H1_plus,
                        H2c_plus=H2c_plus,
                        H1_minus=np.conj(H2c_plus),
                        H2c_minus=np.conj(H1_plus),
                        Iplus=Iplus,
                        Iminus=np.conj(Iplus),
                    )
        if values is None:
            values = self._build_propagator(key)
            if path is not None:
                np.savez_compressed(
                    path,
                    metadata=json.dumps(self.cache_metadata(key), sort_keys=True),
                    H1_plus=values.H1_plus,
                    H2c_plus=values.H2c_plus,
                    Iplus=values.Iplus,
                )
        self._propagators[key] = values
        return values

    def integrate(self, f: ComplexArray, pm: PM = "+") -> complex:
        z = self.z_plus if pm == "+" else self.z_minus
        return complex(np.trapz(np.asarray(f, dtype=np.complex128), z))

    def I(self, pm: PM, momentum_ratio: float = 1.0) -> ComplexArray:
        values = self.propagator(momentum_ratio)
        return values.Iplus if pm == "+" else values.Iminus


def pm_sign(pm: PM) -> int:
    return +1 if pm == "+" else -1


# ---------------------------------------------------------------------------
# Fig. 2: squeezed bispectrum, Eqs. (21)-(23)
# ---------------------------------------------------------------------------

def s_pm(grid: MixedGrid, pm: PM) -> complex:
    sig = pm_sign(pm)
    nu = grid.nu
    integral = grid.integrate(grid.Iplus**2 * cpow(grid.z_plus, -2.5 + sig * nu), "+")
    coeff = -(
        (np.pi ** 2)
        * grid.Cnu
        * np.sin(np.pi / 4 + sig * np.pi * nu / 2)
        / (cpow(2.0 + 0j, 8 + sig * nu) * complex(mp.gamma(1 + sig * nu)) * np.sin(sig * np.pi * nu))
    )
    return complex(coeff * integral)


def squeezed_coefficients(grid: MixedGrid) -> Tuple[float, float, complex, complex]:
    sp = s_pm(grid, "+")
    sm = s_pm(grid, "-")
    s1 = float(np.real(sp + sm))
    s2 = float(np.imag(sm - sp))
    return s1, s2, sp, sm


def fig2_y(x_hard_over_soft: np.ndarray, nutilde: float, s1: float, s2: float) -> np.ndarray:
    # x = k3/k1, and the plotted quantity is x*S_clock.
    x = np.asarray(x_hard_over_soft, dtype=float)
    log_soft = -np.log(x)  # log(k1/k3)
    return np.sqrt(x) * (s1 * np.sin(nutilde * log_soft) + s2 * np.cos(nutilde * log_soft))


# ---------------------------------------------------------------------------
# Fig. 5: triangular trispectrum, Eqs. (33)-(37)
# ---------------------------------------------------------------------------

def triangular_contact_pm(grid: MixedGrid, pm: PM, hard_frac: float = 1 / 3) -> complex:
    sig = pm_sign(pm)
    nu = grid.nu
    I_hard = grid.I("+", hard_frac)
    integrand = I_hard**3 * cpow(grid.z_plus, -2.5 + sig * nu)
    return grid.integrate(integrand, "+")


def triangular_exchange_contour_terms(
    *,
    z_plus: ComplexArray,
    z_minus: ComplexArray,
    power: complex,
    Iplus: ComplexArray,
    Iminus: ComplexArray,
    H1_plus: ComplexArray,
    H2c_plus: ComplexArray,
    H1_minus: ComplexArray,
) -> Tuple[complex, complex]:
    """O(N) Ref. 1803 Eq. (36) on branch-correct Wick contours."""
    Jpp = cpow(z_plus, -2.5) * Iplus**2 * H1_plus
    Jpm = cpow(z_plus, -2.5) * Iplus**2 * H2c_plus
    q_plus = cpow(z_plus, power) * Iplus * H1_plus
    q_minus = cpow(z_minus, power) * Iminus * H1_minus

    ordered_integrand = (
        cpow(z_plus, power) * Iplus * H2c_plus * cumtrapz_complex(Jpp, z_plus)
        + Jpm * cumtrapz_complex(q_plus, z_plus)
    )
    ordered = np.trapz(ordered_integrand, z_plus)
    factorized = np.trapz(Jpm, z_plus) * np.trapz(q_minus, z_minus)
    return complex(ordered), complex(factorized)


def triangular_exchange_one_channel_pm(
    grid: MixedGrid,
    pm: PM,
    hard_frac: float = 1 / 3,
) -> complex:
    """Eq. (36) for one channel in the symmetric hard configuration.

    The Fig. 5 benchmark has k2≈k3≈k4, so s, t and u channels are identical.
    This function computes I_s^±; the final coefficient uses 3*I_s^±.
    """
    sig = pm_sign(pm)
    nu = grid.nu

    integral_T, integral_all = triangular_exchange_contour_terms(
        z_plus=grid.z_plus,
        z_minus=grid.z_minus,
        power=-1 + sig * nu,
        Iplus=grid.Iplus,
        Iminus=grid.Iminus,
        H1_plus=grid.H1_plus,
        H2c_plus=grid.H2c_plus,
        H1_minus=grid.H1_minus,
    )

    return complex((hard_frac ** (1.5 - sig * nu)) * np.exp(-np.pi * grid.nutilde) * (integral_T - integral_all))


def t_pm(
    grid: MixedGrid,
    pm: PM,
    lambda3_over_H: float = 1.0,
    lambda4: float = math.pi / 4,
    hard_frac: float = 1 / 3,
) -> complex:
    sig = pm_sign(pm)
    nu = grid.nu
    Ic = triangular_contact_pm(grid, pm, hard_frac)
    Is = triangular_exchange_one_channel_pm(grid, pm, hard_frac)
    exch = 3.0 * Is
    pref = (
        cpow(2.0 + 0j, 1 - sig * nu)
        * np.sin(np.pi / 4 + sig * np.pi * nu / 2)
        * grid.Cnu
        / (complex(mp.gamma(1 + sig * nu)) * np.sin(sig * np.pi * nu))
    )
    return complex(pref * ((-4j * lambda4 / np.pi) * Ic + (lambda3_over_H ** 2) * exch))


def triangular_coefficients(
    grid: MixedGrid,
    lambda3_over_H: float = 1.0,
    lambda4: float = math.pi / 4,
    hard_frac: float = 1 / 3,
) -> Tuple[float, float, complex, complex]:
    tp = t_pm(grid, "+", lambda3_over_H, lambda4, hard_frac)
    tm = t_pm(grid, "-", lambda3_over_H, lambda4, hard_frac)
    t1 = float(np.real(tp + tm))
    t2 = float(np.imag(tm - tp))
    return t1, t2, tp, tm


def fig5_y_from_complex_diagnostic(x_K_over_k1: np.ndarray, nutilde: float, t_plus: complex, t_minus: complex) -> np.ndarray:
    """Diagnostic complex-form curve; not used for the main Ref. 1803 printed-convention plots.

    For y=k1/K=1/x and nu=i*nutilde,
      x^2 Re[t_+ y^(3/2+nu)+t_- y^(3/2-nu)]
        = sqrt(x) Re[t_+ exp(-i*nutilde*log x)+t_- exp(+i*nutilde*log x)].
    """
    x = np.asarray(x_K_over_k1, dtype=float)
    phase = nutilde * np.log(x)
    return np.sqrt(x) * np.real(t_plus * np.exp(-1j * phase) + t_minus * np.exp(+1j * phase))


def fig5_y_printed_sincos(x_K_over_k1: np.ndarray, nutilde: float, t1: float, t2: float) -> np.ndarray:
    """Ref. 1803 printed convention, Eq. (37): t1*sin+t2*cos."""
    x = np.asarray(x_K_over_k1, dtype=float)
    log_soft = -np.log(x)  # log(k1/K)
    return np.sqrt(x) * (t1 * np.sin(nutilde * log_soft) + t2 * np.cos(nutilde * log_soft))


# ---------------------------------------------------------------------------
# Fig. 8: collapsed trispectrum, Eqs. (39)-(41)
# ---------------------------------------------------------------------------

def c_pm(grid: MixedGrid, pm: PM, k1_over_k3: float = 1.0) -> complex:
    sig = pm_sign(pm)
    nu = grid.nu
    power = -2.5 + sig * nu
    plus_integral = grid.integrate(cpow(grid.z_plus, power) * grid.Iplus**2, "+")
    minus_second = grid.integrate(cpow(grid.z_minus, power) * grid.Iminus**2, "-")
    branch_difference = plus_integral - minus_second
    a = 2.0 + k1_over_k3 + 1.0 / k1_over_k3
    pref = (complex(mp.gamma(-sig * nu)) ** 2) / (cpow(2.0 + 0j, 5 - sig * nu) * (np.pi ** 2))
    pref *= cpow(a + 0j, -2.5 + sig * nu)
    return complex(pref * plus_integral * branch_difference)


def collapsed_coefficients(
    grid: MixedGrid,
    k1_over_k3: float = 1.0,
) -> Tuple[float, float, complex, complex]:
    """Direct Ref. 1803 Eq. (40) coefficients on conjugate Wick contours."""
    cp = c_pm(grid, "+", k1_over_k3)
    cm = c_pm(grid, "-", k1_over_k3)
    c1 = float(np.real(cp + cm))
    c2 = float(np.imag(cm - cp))
    return c1, c2, cp, cm


def fig8_y_from_complex_diagnostic(x_K_over_ks: np.ndarray, nutilde: float, c_plus: complex, c_minus: complex) -> np.ndarray:
    """Diagnostic complex-form curve; not used for the main Ref. 1803 printed-convention plots.

    For y=ks/K=1/x and nu=i*nutilde,
      Re[c_+ y^(2nu)+c_- y^(-2nu)]
        = Re[c_+ exp(-2i*nutilde*log x)+c_- exp(+2i*nutilde*log x)].
    """
    x = np.asarray(x_K_over_ks, dtype=float)
    phase = 2 * nutilde * np.log(x)
    return np.real(c_plus * np.exp(-1j * phase) + c_minus * np.exp(+1j * phase))


def fig8_y_printed_sincos(x_K_over_ks: np.ndarray, nutilde: float, c1: float, c2: float) -> np.ndarray:
    """Ref. 1803 printed convention, Eq. (41): c1*sin+c2*cos."""
    x = np.asarray(x_K_over_ks, dtype=float)
    log_soft = -np.log(x)  # log(ks/K)
    return c1 * np.sin(2 * nutilde * log_soft) + c2 * np.cos(2 * nutilde * log_soft)



# ---------------------------------------------------------------------------
# Article Eq. (3.12), (3.27), (3.34): full correlators along benchmark soft trajectories
# ---------------------------------------------------------------------------

def safe_panel_normalize(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return y/max(|y|) and the max-absolute normalization scale."""
    y = np.asarray(y, dtype=float)
    scale = float(np.nanmax(np.abs(y)))
    if np.isfinite(scale) and scale > 0:
        return y / scale, scale
    return y, 1.0


def cclock_bispectrum_kernel_from_ratio(
    x_k3_over_k1: np.ndarray,
    nutilde: float,
    s1: float,
    s2: float,
) -> np.ndarray:
    """Article Eq. (3.13), evaluated on k2=k3=1 and x=k3/k1.

    C_clock=(k1/k3)^{1/2}[s1 sin(nu log(k1/k3))+s2 cos(nu log(k1/k3))].
    """
    x = np.asarray(x_k3_over_k1, dtype=float)
    log_soft = -np.log(x)  # log(k1/k3)
    return x ** (-0.5) * (s1 * np.sin(nutilde * log_soft) + s2 * np.cos(nutilde * log_soft))


def ctriangle_kernel_from_ratio(
    x_K_over_k1: np.ndarray,
    nutilde: float,
    t1: float,
    t2: float,
) -> np.ndarray:
    """Article Eq. (3.26), evaluated on x=K/k1 using Ref. 1803 printed convention."""
    x = np.asarray(x_K_over_k1, dtype=float)
    log_soft = -np.log(x)  # log(k1/K)
    return x ** (-1.5) * (t1 * np.sin(nutilde * log_soft) + t2 * np.cos(nutilde * log_soft))


def ccollapsed_kernel_from_ratio(
    x_K_over_ks: np.ndarray,
    nutilde: float,
    c1: float,
    c2: float,
) -> np.ndarray:
    """Article Eq. (3.33), evaluated on x=K/ks using Ref. 1803 printed convention."""
    x = np.asarray(x_K_over_ks, dtype=float)
    log_soft = -np.log(x)  # log(ks/K)
    return c1 * np.sin(2.0 * nutilde * log_soft) + c2 * np.cos(2.0 * nutilde * log_soft)


def Bzeta_clock_eq312(
    x_k3_over_k1: np.ndarray,
    nutilde: float,
    s1: float,
    s2: float,
    power_area: float,
    lambda2_over_H: float,
    lambda3_over_H: float,
    hard_k3: float = 1.0,
) -> np.ndarray:
    """Full bispectrum clock correlator in article Eq. (3.12).

    Benchmark trajectory:
      k3=k2=hard_k3, k1=k3/x, x=k3/k1.
    """
    x = np.asarray(x_k3_over_k1, dtype=float)
    k3 = hard_k3
    k2 = hard_k3
    k1 = hard_k3 / x
    C = cclock_bispectrum_kernel_from_ratio(x, nutilde, s1, s2)
    pref = (2.0 * np.pi) ** 4 * np.pi ** 5 / 64.0
    pref *= power_area ** 1.5 * lambda2_over_H ** 3 * lambda3_over_H
    return pref * C / (k1 * k2 * k3) ** 2


def Tzeta_triangle_eq327(
    x_K_over_k1: np.ndarray,
    nutilde: float,
    t1: float,
    t2: float,
    power_area: float,
    lambda2_over_H: float,
    hard_frac: float = 1.0 / 3.0,
    K_total: float = 1.0,
) -> np.ndarray:
    """Full triangular trispectrum clock correlator in article Eq. (3.27).

    Benchmark trajectory:
      K=K_total, k1=K/x, k2=k3=k4=hard_frac*K.
    This is the ordered external-soft approximation used for the kernel.
    """
    x = np.asarray(x_K_over_k1, dtype=float)
    K = K_total
    k1 = K / x
    k2 = hard_frac * K
    k3 = hard_frac * K
    k4 = hard_frac * K
    C = ctriangle_kernel_from_ratio(x, nutilde, t1, t2)
    pref = (2.0 * np.pi) ** 6 * np.pi ** 3 / (2.0 ** 15)
    pref *= power_area ** 2 * lambda2_over_H ** 4
    return pref * K ** 3 * C / (k1 * k2 * k3 * k4) ** 3


def Tzeta_collapsed_eq334(
    x_K_over_ks: np.ndarray,
    nutilde: float,
    c1: float,
    c2: float,
    power_area: float,
    lambda2_over_H: float,
    lambda3_over_H: float,
    hard_k: float = 0.25,
    K_total: float = 1.0,
) -> np.ndarray:
    """Full collapsed trispectrum clock correlator in article Eq. (3.34).

    Benchmark trajectory:
      K=K_total, ks=K/x, and k1=k2=k3=k4=hard_k.
    With hard_k=1/4 and K=1, the hard momenta are normalized consistently.
    """
    x = np.asarray(x_K_over_ks, dtype=float)
    K = K_total
    k1 = hard_k
    k2 = hard_k
    k3 = hard_k
    k4 = hard_k
    C = ccollapsed_kernel_from_ratio(x, nutilde, c1, c2)
    pref = (2.0 * np.pi) ** 6 * np.pi ** 3 / (2.0 ** 15)
    pref *= power_area ** 2 * lambda2_over_H ** 4 * lambda3_over_H ** 2
    return pref * K ** 3 * C / (k1 * k2 * k3 * k4) ** 3


def plot_article_equations_3x4(args: argparse.Namespace, outdir: Path) -> Dict[str, Dict[str, object]]:
    """Make one 4-row x 3-column figure for article Eqs. (3.12), (3.27), (3.34)."""
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_matplotlib_cache(args, outdir)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = np.logspace(args.xmin_plot_exp, args.xmax_plot_exp, args.npoints)
    fig, axes = plt.subplots(
        len(args.nus),
        3,
        figsize=(11.4, 2.55 * len(args.nus)),
        sharex="col",
        squeeze=False,
    )
    column_colors = getattr(
        args,
        "fig1_column_colors",
        {
            "squeezed": "#1f5a96",
            "triangular": "#2a8c55",
            "collapsed": "#7b4ba0",
        },
    )
    column_color_values = [
        column_colors["squeezed"],
        column_colors["triangular"],
        column_colors["collapsed"],
    ]

    # Publication-facing labels: no equation numbers are printed on the figure.
    col_titles = [
        r"Squeezed bispectrum",
        r"Triangular trispectrum",
        r"Collapsed trispectrum",
    ]
    xlabels = [r"$k_3/k_1$", r"$K/k_1$", r"$K/k_s$"]
    ylabels = [
        r"$B_{\zeta,\rm clock}$",
        r"$T_{\zeta,\triangle}$",
        r"$T_{\zeta,\rm coll}$",
    ]

    results: Dict[str, Dict[str, object]] = {
        "metadata": {
            "formula_version": CLOCK_FORMULA_VERSION,
            "coefficient_method": "direct half-way shifted-Wick contour integration",
            "article_equations": ["3.12", "3.27", "3.34"],
            "layout": f"{len(args.nus)} rows for nu_tilde={list(args.nus)}; 3 columns for B, triangular T, collapsed T; physical prefactors included",
            "power_area": float(args.power_area),
            "power_normalization": "area_in_dlnk",
            "lambda2_over_H": float(args.lambda2_over_H),
            "lambda3_over_H": float(args.lambda3_over_H),
            "lambda4": float(args.lambda4),
            "hard_frac_triangle": float(args.hard_frac),
            "hard_k_collapsed": float(args.hard_k_collapsed),
            "K_total": float(args.K_total),
            "hard_k3_bispectrum": float(args.hard_k3_bispectrum),
            "wick_pivot": float(args.wick_pivot),
            "wick_depth": float(args.wick_depth),
            "tail_depth": float(args.tail_depth),
            "real_step": float(args.real_step),
            "vertical_step": float(args.vertical_step),
            "nlog": int(args.nlog),
            "dps": int(args.dps),
            "npoints": int(args.npoints),
            "x_range": [float(10 ** args.xmin_plot_exp), float(10 ** args.xmax_plot_exp)],
            "normalized_per_panel": bool(args.normalize_panels),
            "column_colors": dict(column_colors),
        }
    }

    for row, nt in enumerate(args.nus):
        print(f"[compute] nu_tilde={nt:g}")
        grid = build_grid(nt, args)

        s1, s2, sp, sm = squeezed_coefficients(grid)
        t1, t2, tp, tm = triangular_coefficients(
            grid,
            lambda3_over_H=args.lambda3_over_H,
            lambda4=args.lambda4,
            hard_frac=args.hard_frac,
        )
        c1, c2, cp, cm = collapsed_coefficients(grid, k1_over_k3=args.k1_over_k3)

        yB = Bzeta_clock_eq312(
            x, nt, s1, s2,
            power_area=args.power_area,
            lambda2_over_H=args.lambda2_over_H,
            lambda3_over_H=args.lambda3_over_H,
            hard_k3=args.hard_k3_bispectrum,
        )
        yTtri = Tzeta_triangle_eq327(
            x, nt, t1, t2,
            power_area=args.power_area,
            lambda2_over_H=args.lambda2_over_H,
            hard_frac=args.hard_frac,
            K_total=args.K_total,
        )
        yTcoll = Tzeta_collapsed_eq334(
            x, nt, c1, c2,
            power_area=args.power_area,
            lambda2_over_H=args.lambda2_over_H,
            lambda3_over_H=args.lambda3_over_H,
            hard_k=args.hard_k_collapsed,
            K_total=args.K_total,
        )

        ys = [yB, yTtri, yTcoll]
        scales = []
        if args.normalize_panels:
            ys_plot = []
            for y in ys:
                yn, scale = safe_panel_normalize(y)
                ys_plot.append(yn)
                scales.append(scale)
        else:
            ys_plot = ys
            scales = [1.0, 1.0, 1.0]

        for col, (ax, y, title, xlabel, ylabel) in enumerate(zip(axes[row], ys_plot, col_titles, xlabels, ylabels)):
            ax.set_axisbelow(True)
            ax.grid(True, which="major", color="0.88", lw=0.45)
            ax.grid(True, which="minor", axis="x", color="0.93", lw=0.30)
            ax.plot(x, y, color=column_color_values[col], lw=1.35)
            ax.axhline(0.0, color="0.28", lw=0.85, zorder=1)
            ax.set_xscale("log")
            if args.normalize_panels:
                ax.set_ylim(-1.08, 1.08)
                ax.set_yticks([-1.0, 0.0, 1.0])
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
            ax.tick_params(direction="in", which="both", top=True, right=True, length=3.0, width=0.7)
            ax.tick_params(which="minor", length=1.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
            if row == 0:
                ax.set_title(title, fontsize=10, pad=4)
            if row == len(args.nus) - 1:
                ax.set_xlabel(xlabel, fontsize=10)
            if col == 0:
                ax.text(
                    -0.30,
                    0.50,
                    rf"$\widetilde{{\nu}}={nt:g}$",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    clip_on=False,
                )
                ax.set_ylabel(r"", fontsize=9)
            else:
                ax.tick_params(axis="y", labelleft=False)

        results[str(nt)] = {
            "s1": float(s1),
            "s2": float(s2),
            "s_plus_re": float(np.real(sp)),
            "s_plus_im": float(np.imag(sp)),
            "s_minus_re": float(np.real(sm)),
            "s_minus_im": float(np.imag(sm)),
            "t1": float(t1),
            "t2": float(t2),
            "t_plus_re": float(np.real(tp)),
            "t_plus_im": float(np.imag(tp)),
            "t_minus_re": float(np.real(tm)),
            "t_minus_im": float(np.imag(tm)),
            "c1": float(c1),
            "c2": float(c2),
            "c_plus_re": float(np.real(cp)),
            "c_plus_im": float(np.imag(cp)),
            "c_minus_re": float(np.real(cm)),
            "c_minus_im": float(np.imag(cm)),
            "coefficient_method": "direct half-way shifted-Wick contour integration",
            "panel_normalization": {
                "Eq3.12_Bzeta_max_abs": float(scales[0]),
                "Eq3.27_Ttriangle_max_abs": float(scales[1]),
                "Eq3.34_Tcollapsed_max_abs": float(scales[2]),
            },
        }

    # No suptitle: this figure is intended for direct paper inclusion.
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.075, top=0.94, hspace=0.34, wspace=0.28)

    fig_path = outdir / "qsfi_clock_correlators_4x3.png"
    pdf_path = outdir / "qsfi_clock_correlators_4x3.pdf"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    json_path = outdir / "qsfi_clock_correlators_coefficients_and_normalizations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[saved] {fig_path}")
    print(f"[saved] {pdf_path}")
    print(f"[saved] {json_path}")
    return results


def full_momentum_settings_from_args(args: argparse.Namespace) -> FullMomentumIntegrationSettings:
    """Build the complete-shape PBH settings from a figure namespace."""
    return FullMomentumIntegrationSettings(
        sound_speed=sound_speed_from_args(args),
        lognormal_delta=float(args.lognormal_delta),
        k_peak_r=float(args.k_peak_r),
        lognormal_sigma_range=float(args.full_momentum_sigma_range),
        power_area=float(args.power_area),
        lambda2_over_H=float(args.lambda2_over_H),
        ratio_floor=float(args.full_ratio_floor),
        ratio_nodes=int(args.full_ratio_nodes),
        radial_sampling=str(args.radial_sampling),
        sobol_power_bispectrum=int(args.sobol_power_bispectrum),
        sobol_power_contact=int(args.sobol_power_contact),
        sobol_power_exchange=int(args.sobol_power_exchange),
        sobol_replicates=int(args.sobol_replicates),
        sobol_min_replicates_four_point=int(args.sobol_min_replicates_four_point),
        sobol_max_replicates_four_point=int(args.sobol_max_replicates_four_point),
        sobol_relative_error_target_bispectrum=float(
            args.sobol_relative_error_target_bispectrum
        ),
        sobol_relative_error_target_four_point=float(
            args.sobol_relative_error_target_four_point
        ),
        sobol_max_power_increment_four_point=int(
            args.sobol_max_power_increment_four_point
        ),
        sobol_seed=int(args.sobol_seed),
    )


def heatmap_norm(values: np.ndarray):
    import matplotlib.colors as mcolors

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=-1.0, vmax=1.0)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=max(1e-30, abs(vmax) * 1e-12)):
        pad = max(abs(vmax) * 0.05, 1e-12)
        return mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)
    if vmin < 0.0 < vmax:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def draw_s3_lambda3_figure(
    values: np.ndarray,
    lambda3: np.ndarray,
    nus: Iterable[float],
    args: argparse.Namespace,
    outdir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), sharex=True, squeeze=False)
    color = "#1f5a96"
    for idx, (ax, nt) in enumerate(zip(axes.ravel(), nus)):
        y = np.asarray(values[idx], dtype=float)
        ax.plot(lambda3, y, color=color, lw=1.35)
        ax.axhline(0.0, color="0.55", lw=0.7, zorder=0)
        ax.axvline(0.0, color="0.82", lw=0.6, zorder=0)
        ax.set_title(rf"$\widetilde{{\nu}}={nt:g}$")
        ax.tick_params(direction="in", which="both", top=True, right=True, length=3.0, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        if idx // 2 == 1:
            ax.set_xlabel(r"$\lambda_3/H$")
        if idx % 2 == 0:
            ax.set_ylabel(r"$S_3^{\rm QSFI}$")

    fig.tight_layout(w_pad=0.55, h_pad=0.55)
    base = outdir / "qsfi_pbh_S3_lambda3_scan"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"[saved] {base.with_suffix('.pdf')}")
    print(f"[saved] {base.with_suffix('.png')}")
    return base.with_suffix(".pdf")


def draw_cumulant_heatmap_figure(
    field: str,
    values: np.ndarray,
    qng_values: np.ndarray,
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    nus: Iterable[float],
    args: argparse.Namespace,
    outdir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6), sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.88")
    xgrid, ygrid = np.meshgrid(lambda3, lambda4)
    label = r"$S_3^{\rm QSFI}$" if field == "S3" else r"$S_4^{\rm QSFI}$"

    for idx, (ax, nt) in enumerate(zip(axes.ravel(), nus)):
        panel = np.array(values[idx], dtype=float)
        valid = qng_values[idx] <= args.qng_max
        panel_masked = np.ma.array(panel, mask=~valid)
        mesh = ax.pcolormesh(
            xgrid,
            ygrid,
            panel_masked,
            shading="auto",
            cmap=cmap,
            norm=heatmap_norm(panel_masked),
        )
        if np.any(~valid):
            ax.contourf(
                xgrid,
                ygrid,
                ~valid,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
            )
            ax.contour(
                xgrid,
                ygrid,
                qng_values[idx],
                levels=[args.qng_max],
                colors="0.10",
                linewidths=0.6,
            )
        if np.nanmin(panel) < 0.0 < np.nanmax(panel):
            ax.contour(xgrid, ygrid, panel, levels=[0.0], colors="0.15", linewidths=0.55)

        ax.set_title(rf"$\widetilde{{\nu}}={nt:g}$")
        ax.tick_params(direction="in", which="both", top=True, right=True, length=3.0, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        if idx // 2 == 1:
            ax.set_xlabel(r"$\lambda_3/H$")
        if idx % 2 == 0:
            ax.set_ylabel(r"$\lambda_4$")

        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar.formatter = formatter
        cbar.set_label(label, labelpad=2)
        cbar.ax.tick_params(labelsize=7, direction="in", length=2.5, width=0.6)
        cbar.update_ticks()

    fig.tight_layout(w_pad=0.35, h_pad=0.55)
    base = outdir / ("qsfi_pbh_S3_heatmaps" if field == "S3" else "qsfi_pbh_S4_heatmaps")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"[saved] {base.with_suffix('.pdf')}")
    print(f"[saved] {base.with_suffix('.png')}")
    return base.with_suffix(".pdf")


def plot_cumulant_heatmaps(args: argparse.Namespace, outdir: Path) -> Dict[str, object]:
    """Generate cumulants from the complete three- and four-point shapes."""
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_matplotlib_cache(args, outdir)

    lambda3 = np.linspace(args.lambda3_min, args.lambda3_max, args.nscan)
    lambda4 = np.linspace(args.lambda4_min, args.lambda4_max, args.nscan)
    l3_grid, l4_grid = np.meshgrid(lambda3, lambda4)

    s3_lines = []
    s3_grid_panels = []
    s4_panels = []
    qng_panels = []
    coefficient_records: Dict[str, Dict[str, float]] = {}
    sampling_precision_accepted = True
    full_settings = full_momentum_settings_from_args(args)
    checkpoint_root = Path(args.cache_dir) if args.cache_dir else outdir / "cache"
    checkpoint_tag = full_momentum_checkpoint_tag(args)

    for nt in args.nus:
        print(f"[compute full-momentum cumulants] nu_tilde={nt:g}")
        grid = build_grid(nt, args)
        checkpoint_dir = (
            checkpoint_root
            / "full_momentum_qmc"
            / FORMULA_VERSION
            / checkpoint_tag
            / f"nu_{float(nt):g}"
        )
        coeff = project_with_mixed_grid(
            grid,
            nutilde=float(nt),
            settings=full_settings,
            checkpoint_dir=checkpoint_dir,
        )
        sampling_precision_accepted = bool(
            sampling_precision_accepted and coeff["sampling_precision_accepted"]
        )

        s3_line = coeff["S3_per_lambda3_over_H"] * lambda3
        s3_grid = coeff["S3_per_lambda3_over_H"] * l3_grid
        s4_l3sq_coeff = coeff["S4_per_lambda3_over_H_squared_exchange"]
        s4 = coeff["S4_per_lambda4_contact"] * l4_grid + s4_l3sq_coeff * (l3_grid ** 2)
        qng = qng_from_cumulants(s3_grid, s4)

        s3_lines.append(s3_line)
        s3_grid_panels.append(s3_grid)
        s4_panels.append(s4)
        qng_panels.append(qng)
        coefficient_records[str(nt)] = {
            **coeff,
            "coefficient_method": "complete Eq. (20)/(30)-(32) momentum shapes with Sobol PBH projection",
            "S3_min": float(np.nanmin(s3_line)),
            "S3_max": float(np.nanmax(s3_line)),
            "S4_min": float(np.nanmin(s4)),
            "S4_max": float(np.nanmax(s4)),
            "QNG_min": float(np.nanmin(qng)),
            "QNG_max": float(np.nanmax(qng)),
            "QNG_abs_max": float(np.nanmax(np.abs(qng))),
            "QNG_bound_satisfied": bool(np.nanmax(qng) <= args.qng_max),
        }

    s3_arr = np.asarray(s3_lines, dtype=float)
    s3_grid_arr = np.asarray(s3_grid_panels, dtype=float)
    s4_arr = np.asarray(s4_panels, dtype=float)
    qng_arr = np.asarray(qng_panels, dtype=float)

    draw_s3_lambda3_figure(s3_arr, lambda3, args.nus, args, outdir)
    draw_cumulant_heatmap_figure("S4", s4_arr, qng_arr, lambda3, lambda4, args.nus, args, outdir)

    npz_path = outdir / "qsfi_pbh_cumulant_heatmaps_data.npz"
    np.savez_compressed(
        npz_path,
        nus=np.asarray(args.nus, dtype=float),
        lambda3_over_H=lambda3,
        lambda4=lambda4,
        S3=s3_arr,
        S3_grid=s3_grid_arr,
        S4=s4_arr,
        QNG=qng_arr,
    )
    print(f"[saved] {npz_path}")

    summary = {
        "metadata": {
            "formula_version": FORMULA_VERSION,
            "coefficient_method": "complete Eq. (20)/(30)-(32) momentum shapes with Sobol PBH projection",
            "projection": "full_momentum_qmc",
            **sound_speed_from_args(args).as_dict(),
            "sampling_precision_accepted": sampling_precision_accepted,
            "numerical_acceptance": (
                "sampling_target_met_pending_independent_convergence"
                if sampling_precision_accepted
                else "provisional_sampling_error"
            ),
            "power_area": float(args.power_area),
            "peak_height": peak_height_from_area(
                float(args.power_area),
                float(args.lognormal_delta),
            ),
            "power_normalization": str(args.power_normalization),
            "lognormal_delta": float(args.lognormal_delta),
            "k_peak_r": float(args.k_peak_r),
            "variance_response": density_variance_response(float(args.lognormal_delta), float(args.k_peak_r)),
            "sigma2": float(args.sigma2) if args.sigma2 is not None else float(args.power_area) * density_variance_response(float(args.lognormal_delta), float(args.k_peak_r)),
            "full_momentum_sigma_range": float(args.full_momentum_sigma_range),
            "full_ratio_floor": float(args.full_ratio_floor),
            "full_ratio_nodes": int(args.full_ratio_nodes),
            "radial_sampling": str(args.radial_sampling),
            "sobol_power_bispectrum": int(args.sobol_power_bispectrum),
            "sobol_power_contact": int(args.sobol_power_contact),
            "sobol_power_exchange": int(args.sobol_power_exchange),
            "sobol_replicates": int(args.sobol_replicates),
            "sobol_min_replicates_four_point": int(args.sobol_min_replicates_four_point),
            "sobol_max_replicates_four_point": int(args.sobol_max_replicates_four_point),
            "sobol_relative_error_target_bispectrum": float(
                args.sobol_relative_error_target_bispectrum
            ),
            "sobol_relative_error_target_four_point": float(
                args.sobol_relative_error_target_four_point
            ),
            "sobol_max_power_increment_four_point": int(
                args.sobol_max_power_increment_four_point
            ),
            "sobol_seed": int(args.sobol_seed),
            "lambda2_over_H": float(args.lambda2_over_H),
            "lambda3_over_H_range": [float(args.lambda3_min), float(args.lambda3_max)],
            "lambda4_range": [float(args.lambda4_min), float(args.lambda4_max)],
            "nscan": int(args.nscan),
            "qng_max": float(args.qng_max),
            "wick_pivot": float(args.wick_pivot),
            "wick_depth": float(args.wick_depth),
            "tail_depth": float(args.tail_depth),
            "real_step": float(args.real_step),
            "vertical_step": float(args.vertical_step),
            "nlog": int(args.nlog),
            "dps": int(args.dps),
        },
        "coefficients": coefficient_records,
    }
    json_path = outdir / "qsfi_pbh_cumulant_heatmaps_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"[saved] {json_path}")
    return summary


# ---------------------------------------------------------------------------
# Plotting and CLI
# ---------------------------------------------------------------------------

def maybe_normalize(y: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return y
    m = np.nanmax(np.abs(y))
    return y / m if np.isfinite(m) and m > 0 else y


def ensure_matplotlib_cache(args: argparse.Namespace, outdir: Path) -> None:
    if args.cache_dir:
        mpldir = Path(args.cache_dir) / "mplconfig"
    else:
        mpldir = Path(tempfile.gettempdir()) / "qsfi_1803_r1_mplconfig"
    mpldir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpldir))


def build_grid(nt: float, args: argparse.Namespace) -> MixedGrid:
    return MixedGrid(
        nutilde=nt,
        sound_speed=sound_speed_from_args(args),
        pivot=args.wick_pivot,
        depth=args.wick_depth,
        real_step=args.real_step,
        vertical_step=args.vertical_step,
        xmin=args.xmin,
        nlog=args.nlog,
        tail_depth=args.tail_depth,
        dps=args.dps,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        force=args.force,
    )


def curve_from_record(mode: str, x: np.ndarray, nt: float, record: Dict[str, float]) -> np.ndarray:
    if mode == "fig2":
        return fig2_y(x, nt, record["s1"], record["s2"])
    if mode == "fig5":
        tp = record["t_plus_re"] + 1j * record["t_plus_im"]
        tm = record["t_minus_re"] + 1j * record["t_minus_im"]
        return fig5_y_printed_sincos(x, nt, record["t1"], record["t2"])
    if mode == "fig8":
        cp = record["c_plus_re"] + 1j * record["c_plus_im"]
        cm = record["c_minus_re"] + 1j * record["c_minus_im"]
        return fig8_y_printed_sincos(x, nt, record["c1"], record["c2"])
    raise ValueError(mode)


def coefficient_rel_diffs(base: Dict[str, float], refined: Dict[str, float]) -> Dict[str, float]:
    diffs: Dict[str, float] = {}
    for key, base_value in base.items():
        if key not in refined:
            continue
        if not isinstance(base_value, (int, float)) or not isinstance(refined[key], (int, float)):
            continue
        ref_value = refined[key]
        denom = max(abs(ref_value), 1e-300)
        diffs[key] = float(abs(base_value - ref_value) / denom)
    return diffs


def coefficient_abs_diffs(base: Dict[str, float], refined: Dict[str, float]) -> Dict[str, float]:
    diffs: Dict[str, float] = {}
    for key, base_value in base.items():
        if key not in refined:
            continue
        if not isinstance(base_value, (int, float)) or not isinstance(refined[key], (int, float)):
            continue
        diffs[key] = float(abs(base_value - refined[key]))
    return diffs


def complex_pair_convergence_stats(base: Dict[str, float], refined: Dict[str, float], prefix: str) -> Dict[str, float]:
    base_z = complex(base[f"{prefix}_re"], base[f"{prefix}_im"])
    refined_z = complex(refined[f"{prefix}_re"], refined[f"{prefix}_im"])
    abs_diff = abs(base_z - refined_z)
    amp_rel = abs(abs(base_z) - abs(refined_z)) / max(abs(refined_z), 1e-300)
    phase_diff = float("nan") if abs(refined_z) == 0 else float(np.angle(base_z / refined_z))
    return {
        "abs_diff": float(abs_diff),
        "amp_rel_diff": float(amp_rel),
        "phase_diff": phase_diff,
    }


def complex_convergence_stats(base: Dict[str, float], refined: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    prefixes = sorted({key[:-3] for key in base if key.endswith("_re") and f"{key[:-3]}_im" in base})
    for prefix in prefixes:
        if f"{prefix}_re" in refined and f"{prefix}_im" in refined:
            stats[prefix] = complex_pair_convergence_stats(base, refined, prefix)
    return stats


def plot_mode(mode: str, args: argparse.Namespace, outdir: Path) -> Dict[str, Dict[str, float]]:
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_matplotlib_cache(args, outdir)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.logspace(args.xmin_plot_exp, args.xmax_plot_exp, args.npoints)
    fig, axes = plt.subplots(len(args.nus), 1, figsize=(7.2, 1.75 * len(args.nus)), sharex=True)
    if len(args.nus) == 1:
        axes = [axes]
    results: Dict[str, Dict[str, float]] = {}

    for ax, nt in zip(axes, args.nus):
        grid = build_grid(nt, args)
        if mode == "fig2":
            a1, a2, ap, am = squeezed_coefficients(grid)
            y = fig2_y(x, nt, a1, a2)
            ylabel = r"$(k_3/k_1)S_{\rm clock}$"
            xlabel = r"$k_3/k_1$"
            coeff_names = ("s1", "s2", "s_plus", "s_minus")
        elif mode == "fig5":
            a1, a2, ap, am = triangular_coefficients(
                grid,
                lambda3_over_H=args.lambda3_over_H,
                lambda4=args.lambda4,
                hard_frac=args.hard_frac,
            )
            y = fig5_y_printed_sincos(x, nt, a1, a2)
            ylabel = r"$(K/k_1)^2T_{\triangle,\rm clock}$"
            xlabel = r"$K/k_1$"
            coeff_names = ("t1", "t2", "t_plus", "t_minus")
        elif mode == "fig8":
            a1, a2, ap, am = collapsed_coefficients(grid, k1_over_k3=args.k1_over_k3)
            y = fig8_y_printed_sincos(x, nt, a1, a2)
            ylabel = r"$T_{\rm coll,clock}$"
            xlabel = r"$K/k_s$"
            coeff_names = ("c1", "c2", "c_plus", "c_minus")
        else:
            raise ValueError(mode)

        y = maybe_normalize(y, args.normalize)
        ax.plot(
            x,
            y,
            color="black",
            lw=1.4,
            label=rf"$r={grid.sound_speed.ratio:g}$",
        )
        ax.axhline(0.0, color="0.75", lw=0.6)
        ax.set_xscale("log")
        ax.tick_params(axis="y", labelleft=False)
        ax.text(
            0.05,
            0.72,
            rf"$\tilde\nu={nt:g}$",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
        ax.legend(loc="upper center", frameon=False, ncol=1)
        ax.grid(True, which="both", axis="x", alpha=0.25)
        ax.set_ylabel(ylabel)

        results[str(nt)] = {
            coeff_names[0]: float(a1),
            coeff_names[1]: float(a2),
            f"{coeff_names[2]}_re": float(np.real(ap)),
            f"{coeff_names[2]}_im": float(np.imag(ap)),
            f"{coeff_names[3]}_re": float(np.real(am)),
            f"{coeff_names[3]}_im": float(np.imag(am)),
        }

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout(h_pad=0.05)
    path = outdir / f"qsfi_1803_r1_{mode}.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return results


def run_convergence(args: argparse.Namespace, outdir: Path) -> Dict[str, object]:
    base_args = argparse.Namespace(**vars(args))
    refined_args = argparse.Namespace(**vars(args))
    refined_args.wick_pivot = args.conv_wick_pivot
    refined_args.wick_depth = args.conv_wick_depth
    refined_args.tail_depth = args.conv_tail_depth
    refined_args.real_step = args.conv_real_step
    refined_args.vertical_step = args.conv_vertical_step
    refined_args.nlog = args.conv_nlog
    refined_args.dps = args.conv_dps
    refined_args.force = args.force

    modes = ["fig2", "fig5", "fig8"] if args.mode == "all" else [args.mode]
    x = np.logspace(args.xmin_plot_exp, args.xmax_plot_exp, args.npoints)
    report: Dict[str, object] = {
        "base": {
            "wick_pivot": base_args.wick_pivot,
            "wick_depth": base_args.wick_depth,
            "tail_depth": base_args.tail_depth,
            "real_step": base_args.real_step,
            "vertical_step": base_args.vertical_step,
            "nlog": base_args.nlog,
            "dps": base_args.dps,
        },
        "refined": {
            "wick_pivot": refined_args.wick_pivot,
            "wick_depth": refined_args.wick_depth,
            "tail_depth": refined_args.tail_depth,
            "real_step": refined_args.real_step,
            "vertical_step": refined_args.vertical_step,
            "nlog": refined_args.nlog,
            "dps": refined_args.dps,
        },
        "modes": {},
    }

    base_out = outdir / "convergence_base"
    refined_out = outdir / "convergence_refined"
    for mode in modes:
        print(f"[convergence] base {mode}")
        base = plot_mode(mode, base_args, base_out)
        print(f"[convergence] refined {mode}")
        refined = plot_mode(mode, refined_args, refined_out)

        mode_report: Dict[str, object] = {}
        for nt in args.nus:
            key = str(nt)
            y_base = curve_from_record(mode, x, nt, base[key])
            y_refined = curve_from_record(mode, x, nt, refined[key])
            mode_report[key] = {
                "coefficient_rel_diffs": coefficient_rel_diffs(base[key], refined[key]),
                "coefficient_abs_diffs": coefficient_abs_diffs(base[key], refined[key]),
                "complex_coefficients": complex_convergence_stats(base[key], refined[key]),
                "normalized_curve": normalized_curve_stats(y_base, y_refined),
            }
        report["modes"][mode] = mode_report

    with open(outdir / "convergence_r1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"[saved] {outdir / 'convergence_r1.json'}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot article Eqs. (3.12), (3.27), (3.34) in a 4x3 normalized panel layout.")
    p.add_argument("--mode", choices=["article", "cumulants", "fig2", "fig5", "fig8", "all"], default="article", help="Default article makes the 4x3 Eq. (3.12)/(3.27)/(3.34) plot; cumulants makes the S3 line scan and S4 heatmaps.")
    p.add_argument("--nus", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0], help="Values of nu_tilde=sqrt((m/H)^2-9/4).")
    p.add_argument("--wick-pivot", type=float, default=4.0, help="Positive-real turning point of the shifted-Wick contour.")
    p.add_argument("--wick-depth", type=float, default=16.0, help="Vertical depth of the shifted-Wick contour.")
    p.add_argument("--tail-depth", type=float, default=18.0, help="Argument-plane depth used to close the stable propagator tail.")
    p.add_argument("--real-step", type=float, default=0.025, help="Maximum step on the linear positive-real segment.")
    p.add_argument("--vertical-step", type=float, default=0.025, help="Maximum step on the vertical contour segment.")
    p.add_argument("--nlog", type=int, default=500, help="Logarithmic points resolving the origin-to-one segment.")
    p.add_argument("--xmin", type=float, default=1e-6)
    p.add_argument("--dps", type=int, default=50, help="mpmath decimal precision for complex-order Hankel functions.")
    p.add_argument("--c-phi", type=float, default=1.0, help="Adiabatic-mode sound speed, with 0 < c_phi <= 1.")
    p.add_argument("--c-sigma", type=float, default=1.0, help="Massive-mode sound speed, with 0 < c_sigma <= 1.")
    p.add_argument(
        "--sound-speed-i-epsilon",
        type=float,
        default=0.0,
        help="Small lower-half-plane regulator used in r=c_phi/c_sigma.",
    )
    p.add_argument("--lambda3-over-H", type=float, default=1.0, help="Fig. 5 benchmark: lambda_3/H=1.")
    p.add_argument("--lambda4", type=float, default=math.pi / 4, help="Fig. 5 benchmark: lambda_4=pi/4.")
    p.add_argument("--hard-frac", type=float, default=1/3, help="Triangular hard ratios k2/K=k3/K=k4/K in the soft limit.")
    p.add_argument("--k1-over-k3", type=float, default=1.0, help="Collapsed hard configuration benchmark.")
    p.add_argument("--xmin-plot-exp", type=float, default=0.0)
    p.add_argument("--xmax-plot-exp", type=float, default=4.0)
    p.add_argument("--npoints", type=int, default=600)
    p.add_argument("--normalize", action="store_true", help="Normalize each curve by its max absolute value; useful for visual shape-only comparison.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--cache-dir", type=str, default="cache")
    p.add_argument("--force", action="store_true", help="Ignore cached Hankel/mixed-propagator arrays.")
    p.add_argument("--convergence", action="store_true", help="Compare the requested contour with a deeper/finer contour.")
    p.add_argument("--conv-wick-pivot", type=float, default=4.5)
    p.add_argument("--conv-wick-depth", type=float, default=18.0)
    p.add_argument("--conv-tail-depth", type=float, default=20.0)
    p.add_argument("--conv-real-step", type=float, default=0.018)
    p.add_argument("--conv-vertical-step", type=float, default=0.018)
    p.add_argument("--conv-nlog", type=int, default=700)
    p.add_argument("--conv-dps", type=int, default=55)
    p.add_argument("--peak-height", type=float, default=1e-2, help="Primary physical amplitude P_zeta(k_*) of the lognormal spectrum.")
    p.add_argument("--power-area", type=float, default=None, help="Optional integrated-area override. If omitted, A=sqrt(2*pi)*Delta*P_zeta(k_*).")
    p.add_argument("--lognormal-delta", type=float, default=0.5, help="Lognormal width in ln(k/k_*).")
    p.add_argument("--k-peak-r", type=float, default=math.sqrt(2.0), help="Dimensionless peak position k_* R.")
    p.add_argument("--sigma2", type=float, default=None, help="Optional variance override; by default sigma^2=A*C2 for the lognormal spectrum.")
    p.add_argument("--lambda2-over-H", type=float, default=_PRODUCTION_CONFIG.lambda2_over_h, help="Fixed mixing coupling lambda_2/H used in article correlator prefactors.")
    p.add_argument("--hard-k3-bispectrum", type=float, default=1.0, help="Hard scale k2=k3 for Eq. (3.12) benchmark trajectory.")
    p.add_argument("--K-total", type=float, default=1.0, help="Reference total K used for trispectrum benchmark trajectories.")
    p.add_argument("--hard-k-collapsed", type=float, default=0.25, help="Hard external momenta k1=k2=k3=k4 for Eq. (3.34) benchmark trajectory when K=1.")
    p.add_argument("--normalize-panels", dest="normalize_panels", action="store_true", help="Optionally normalize each panel by its max absolute value.")
    p.set_defaults(normalize_panels=False)
    p.add_argument("--lambda3-min", type=float, default=_PRODUCTION_CONFIG.lambda3_min, help="Minimum lambda_3/H for cumulant scans.")
    p.add_argument("--lambda3-max", type=float, default=_PRODUCTION_CONFIG.lambda3_max, help="Maximum lambda_3/H for cumulant scans.")
    p.add_argument("--lambda4-min", type=float, default=_PRODUCTION_CONFIG.lambda4_min, help="Minimum lambda_4 for cumulant heatmaps.")
    p.add_argument("--lambda4-max", type=float, default=_PRODUCTION_CONFIG.lambda4_max, help="Maximum lambda_4 for cumulant heatmaps.")
    p.add_argument("--nscan", type=int, default=_PRODUCTION_CONFIG.lambda3_points, help="Number of points per coupling axis in cumulant heatmaps.")
    p.add_argument("--full-momentum-sigma-range", type=float, default=5.0, help="Lognormal momentum bounds in standard deviations of ln(k/k_*).")
    p.add_argument("--full-ratio-floor", type=float, default=1e-5, help="Smallest tabulated external/internal momentum ratio.")
    p.add_argument("--full-ratio-nodes", type=int, default=128, help="Logarithmic ratio nodes for mixed-propagator interpolation.")
    p.add_argument("--sobol-power-bispectrum", type=int, default=12, help="Use 2^m samples per bispectrum Sobol replicate.")
    p.add_argument("--radial-sampling", choices=("log_uniform", "sqrt_lognormal_importance"), default="sqrt_lognormal_importance", help="Radial momentum proposal; the importance option keeps the exact Jacobian.")
    p.add_argument("--sobol-power-contact", type=int, default=12, help="Initial 2^m samples per contact-trispectrum Sobol replicate.")
    p.add_argument("--sobol-power-exchange", type=int, default=12, help="Initial 2^m samples per exchange-trispectrum Sobol replicate.")
    p.add_argument("--sobol-replicates", type=int, default=4, help="Fixed independent scrambled Sobol replicates for the bispectrum.")
    p.add_argument("--sobol-min-replicates-four-point", type=int, default=6, help="Minimum independent replicas for each four-point Sobol level.")
    p.add_argument("--sobol-max-replicates-four-point", type=int, default=12, help="Maximum independent replicas for each four-point Sobol level.")
    p.add_argument("--sobol-relative-error-target-bispectrum", type=float, default=0.01, help="Recorded relative standard-error target for the fixed bispectrum run.")
    p.add_argument("--sobol-relative-error-target-four-point", type=float, default=0.05, help="Adaptive relative standard-error target for contact and exchange terms.")
    p.add_argument("--sobol-max-power-increment-four-point", type=int, default=3, help="Maximum number of automatic Sobol-level refinements for four-point terms.")
    p.add_argument("--sobol-seed", type=int, default=1729, help="Base seed for reproducible scrambled Sobol replicates.")
    p.add_argument(
        "--qng-max",
        type=float,
        default=QNG_MAX,
        help="Upper bound for Q_NG=S3+delta_c*S4/4.",
    )
    p.add_argument("--zip-clean", action="store_true", help="Write a clean zip of the output directory, excluding OS and matplotlib cache files.")
    p.add_argument("--zip-path", type=str, default=None, help="Optional path for --zip-clean output.")
    return p.parse_args()


def main() -> None:
    # VS Code usage: open this file and click "Run Python File".
    # Default mode="article" produces one 4-row x 3-column figure:
    # rows are nu_tilde=1,2,3,4 and columns are Eqs. (3.12), (3.27), (3.34).
    # Defaults use the shifted-Wick contour and stable Eq. (11) tail form.
    args = parse_args()
    if args.power_area is None:
        args.power_area = area_from_peak_height(
            args.peak_height,
            args.lognormal_delta,
        )
    else:
        args.peak_height = peak_height_from_area(
            args.power_area,
            args.lognormal_delta,
        )
    args.power_normalization = "area_in_dlnk"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "article":
        plot_article_equations_3x4(args, outdir)
        if args.zip_clean:
            zip_path = write_clean_zip(outdir, Path(args.zip_path) if args.zip_path else None)
            print(f"[saved] {zip_path}")
        return

    if args.mode == "cumulants":
        plot_cumulant_heatmaps(args, outdir)
        if args.zip_clean:
            zip_path = write_clean_zip(outdir, Path(args.zip_path) if args.zip_path else None)
            print(f"[saved] {zip_path}")
        return

    # Legacy 1803-style individual panels are still available via --mode fig2/fig5/fig8/all.
    if args.convergence:
        run_convergence(args, outdir)
        if args.zip_clean:
            zip_path = write_clean_zip(outdir, Path(args.zip_path) if args.zip_path else None)
            print(f"[saved] {zip_path}")
        return

    modes = ["fig2", "fig5", "fig8"] if args.mode == "all" else [args.mode]
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in modes:
        print(f"[compute] {mode}")
        all_results[mode] = plot_mode(mode, args, outdir)
        write_coefficients_json(outdir, {mode: all_results[mode]}, args)
        print(f"[saved] {outdir / f'qsfi_1803_r1_{mode}.png'}")
    write_coefficients_json(outdir, all_results, args)
    print(f"[saved] {outdir / 'coefficients_r1.json'}")
    if args.zip_clean:
        zip_path = write_clean_zip(outdir, Path(args.zip_path) if args.zip_path else None)
        print(f"[saved] {zip_path}")
    print("Default article plot: rerun without --mode, or use --mode article.")


if __name__ == "__main__":
    main()
