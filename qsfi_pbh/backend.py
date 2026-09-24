#!/usr/bin/env python3
"""Shared Fig. 2 cumulant data generation and loading utilities."""
from __future__ import annotations

import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from .propagator import FORMULA_VERSION, plot_cumulant_heatmaps
from .collapse import QNG_MAX
from .config import ProductionConfig
from .spectrum import (
    POWER_NORMALIZATION,
    area_from_peak_height,
    density_variance_response,
    peak_height_from_area,
)
from .sound_speed import SOUND_SPEED_CASES, SoundSpeedConfiguration

OUTDIR = PROJECT_DIR / "figures"
CACHE_DIR = PROJECT_DIR / ".cache"
SUMMARY_PATH = OUTDIR / "qsfi_pbh_cumulant_heatmaps_summary.json"
DATA_PATH = OUTDIR / "qsfi_pbh_cumulant_heatmaps_data.npz"

# Benchmark used in the optimized light-clock manuscript figures.
NUS = [0.1, 0.3, 0.5, 1.0]
DEFAULT_SOUND_SPEED = SOUND_SPEED_CASES[1]
LOGNORMAL_DELTA = 0.5
REFERENCE_PEAK_HEIGHT = 1.0e-2
POWER_AREA = area_from_peak_height(REFERENCE_PEAK_HEIGHT, LOGNORMAL_DELTA)
K_PEAK_R = math.sqrt(2.0)
SIGMA2 = None  # None means sigma^2 = POWER_AREA * variance response.
_PRODUCTION_CONFIG = ProductionConfig()
LAMBDA2_OVER_H = _PRODUCTION_CONFIG.lambda2_over_h
LAMBDA3_MIN = _PRODUCTION_CONFIG.lambda3_min
LAMBDA3_MAX = _PRODUCTION_CONFIG.lambda3_max
LAMBDA4_MIN = _PRODUCTION_CONFIG.lambda4_min
LAMBDA4_MAX = _PRODUCTION_CONFIG.lambda4_max
# Staging is square at the cubic-axis count for coefficient diagnostics; final
# public arrays use ProductionConfig.coupling_axes() (2001 x 2001).
PRODUCTION_NSCAN = _PRODUCTION_CONFIG.lambda3_points
# Backward-compatible name for callers that inspect the production setting.
NSCAN = PRODUCTION_NSCAN

# Complete momentum-shape projection. The lognormal range is measured in
# standard deviations of ln(k/k_*). Each stage uses independent scrambled
# Sobol replicates so its numerical uncertainty is recorded explicitly.
FULL_MOMENTUM_SIGMA_RANGE = 5.0
FULL_RATIO_FLOOR = 1.0e-5
FULL_RATIO_NODES = 128
SOBOL_POWER_BISPECTRUM = 12
SOBOL_POWER_CONTACT = 12
SOBOL_POWER_EXCHANGE = 12
SOBOL_REPLICATES = 4
RADIAL_SAMPLING = "sqrt_lognormal_importance"
SOBOL_MIN_REPLICATES_FOUR_POINT = 6
SOBOL_MAX_REPLICATES_FOUR_POINT = 12
SOBOL_RELATIVE_ERROR_TARGET_BISPECTRUM = 0.01
SOBOL_RELATIVE_ERROR_TARGET_FOUR_POINT = 0.05
SOBOL_MAX_POWER_INCREMENT_FOUR_POINT = 3
SOBOL_SEED = 1729

# Numerical settings for the production calculation.
WICK_PIVOT = 4.0
WICK_DEPTH = 16.0
TAIL_DEPTH = 18.0
REAL_STEP = 0.025
VERTICAL_STEP = 0.025
NLOG = 500
DPS = 50
DPI = 300
FORCE_RECOMPUTE = False


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
        print(f"[removed] {path}")
    except FileNotFoundError:
        pass


def cleanup_generated_cumulant_auxiliary_outputs() -> None:
    """Remove legacy combined and backend helper plots after split Fig. 2 runs."""
    for name in [
        "qsfi_pbh_S3_lambda3_scan.pdf",
        "qsfi_pbh_S3_lambda3_scan.png",
        "qsfi_pbh_S3_heatmaps.pdf",
        "qsfi_pbh_S3_heatmaps.png",
        "qsfi_pbh_S4_heatmaps.pdf",
        "qsfi_pbh_S4_heatmaps.png",
        "fig2_qsfi_cumulants.pdf",
        "fig2_qsfi_cumulants.png",
    ]:
        remove_if_exists(OUTDIR / name)


def configure_matplotlib_cache() -> None:
    configured = os.environ.get("MPLCONFIGDIR")
    cache_dir = Path(configured) if configured else PROJECT_DIR / ".cache" / "mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not configured:
        os.environ["MPLCONFIGDIR"] = str(cache_dir)


def coupling_scan_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(LAMBDA3_MIN, LAMBDA3_MAX, PRODUCTION_NSCAN),
        np.linspace(LAMBDA4_MIN, LAMBDA4_MAX, PRODUCTION_NSCAN),
    )


def panel_layout(nus: object) -> tuple[int, int]:
    n_panels = len(nus)  # type: ignore[arg-type]
    if n_panels <= 0:
        raise ValueError("At least one nu_tilde value is required.")
    if n_panels <= 2:
        return 1, n_panels
    return 2, int(math.ceil(n_panels / 2.0))


def make_args(
    force: bool = False,
    *,
    sound_speed: SoundSpeedConfiguration | None = None,
) -> Namespace:
    if not math.isfinite(LAMBDA2_OVER_H) or LAMBDA2_OVER_H <= 0.0:
        raise ValueError("lambda2/H must be finite and positive.")
    configuration = DEFAULT_SOUND_SPEED if sound_speed is None else sound_speed
    if not isinstance(configuration, SoundSpeedConfiguration):
        raise TypeError("sound_speed must be a SoundSpeedConfiguration.")
    return Namespace(
        nus=NUS,
        wick_pivot=WICK_PIVOT,
        wick_depth=WICK_DEPTH,
        tail_depth=TAIL_DEPTH,
        real_step=REAL_STEP,
        vertical_step=VERTICAL_STEP,
        nlog=NLOG,
        xmin=1.0e-6,
        dps=DPS,
        cache_dir=str(CACHE_DIR),
        force=force,
        c_phi=configuration.c_phi,
        c_sigma=configuration.c_sigma,
        sound_speed_ratio=configuration.ratio,
        sound_speed_i_epsilon=configuration.i_epsilon,
        power_area=POWER_AREA,
        power_normalization=POWER_NORMALIZATION,
        lognormal_delta=LOGNORMAL_DELTA,
        k_peak_r=K_PEAK_R,
        sigma2=SIGMA2,
        full_momentum_sigma_range=FULL_MOMENTUM_SIGMA_RANGE,
        full_ratio_floor=FULL_RATIO_FLOOR,
        full_ratio_nodes=FULL_RATIO_NODES,
        radial_sampling=RADIAL_SAMPLING,
        sobol_power_bispectrum=SOBOL_POWER_BISPECTRUM,
        sobol_power_contact=SOBOL_POWER_CONTACT,
        sobol_power_exchange=SOBOL_POWER_EXCHANGE,
        sobol_replicates=SOBOL_REPLICATES,
        sobol_min_replicates_four_point=SOBOL_MIN_REPLICATES_FOUR_POINT,
        sobol_max_replicates_four_point=SOBOL_MAX_REPLICATES_FOUR_POINT,
        sobol_relative_error_target_bispectrum=SOBOL_RELATIVE_ERROR_TARGET_BISPECTRUM,
        sobol_relative_error_target_four_point=SOBOL_RELATIVE_ERROR_TARGET_FOUR_POINT,
        sobol_max_power_increment_four_point=SOBOL_MAX_POWER_INCREMENT_FOUR_POINT,
        sobol_seed=SOBOL_SEED,
        lambda2_over_H=LAMBDA2_OVER_H,
        hard_frac=1.0 / 3.0,
        k1_over_k3=1.0,
        lambda3_min=LAMBDA3_MIN,
        lambda3_max=LAMBDA3_MAX,
        lambda4_min=LAMBDA4_MIN,
        lambda4_max=LAMBDA4_MAX,
        nscan=PRODUCTION_NSCAN,
        qng_max=QNG_MAX,
        dpi=DPI,
    )


def expected_metadata(args: Namespace) -> dict[str, object]:
    response = density_variance_response(float(args.lognormal_delta), float(args.k_peak_r))
    return {
        "formula_version": FORMULA_VERSION,
        "coefficient_method": "complete Eq. (20)/(30)-(32) momentum shapes with Sobol PBH projection",
        "projection": "full_momentum_qmc",
        "sound_speed_ratio": float(args.sound_speed_ratio),
        "c_phi": float(args.c_phi),
        "c_sigma": float(args.c_sigma),
        "sound_speed_i_epsilon": float(args.sound_speed_i_epsilon),
        "peak_height": peak_height_from_area(
            float(args.power_area),
            float(args.lognormal_delta),
        ),
        "power_area": float(args.power_area),
        "power_normalization": str(args.power_normalization),
        "lognormal_delta": float(args.lognormal_delta),
        "k_peak_r": float(args.k_peak_r),
        "variance_response": response,
        "sigma2": float(args.sigma2) if args.sigma2 is not None else float(args.power_area) * response,
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
    }


def _metadata_value_matches(found: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(found), expected, rel_tol=0.0, abs_tol=1.0e-14)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        if not isinstance(found, (list, tuple)) or len(found) != len(expected):
            return False
        return all(_metadata_value_matches(f, e) for f, e in zip(found, expected))
    return found == expected


def cached_cumulant_data_is_current(args: Namespace) -> bool:
    """Return True only when existing files match the current benchmark."""
    if not (SUMMARY_PATH.exists() and DATA_PATH.exists()):
        return False
    try:
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        found_meta = summary.get("metadata", {})
        if not isinstance(found_meta, dict):
            return False
        expected_meta = expected_metadata(args)
        for key, expected in expected_meta.items():
            if key not in found_meta or not _metadata_value_matches(
                found_meta[key], expected
            ):
                print(
                    f"[cache mismatch] metadata field {key!r}: "
                    f"found={found_meta.get(key)!r}, expected={expected!r}"
                )
                return False
        with np.load(DATA_PATH) as data:
            if data["nus"].shape[0] != len(args.nus):
                return False
            if not np.allclose(data["nus"], np.asarray(args.nus, dtype=float), rtol=0.0, atol=1.0e-14):
                return False
            if data["lambda3_over_H"].shape[0] != int(args.nscan):
                return False
            if data["lambda4"].shape[0] != int(args.nscan):
                return False
            if data["S4"].shape != (len(args.nus), int(args.nscan), int(args.nscan)):
                return False
            if data["QNG"].shape != (len(args.nus), int(args.nscan), int(args.nscan)):
                return False
    except Exception as exc:
        print(f"[cache mismatch] could not validate existing cumulant files: {exc}")
        return False
    return True


def ensure_cumulant_data(force: bool | None = None) -> None:
    args = make_args(force=bool(FORCE_RECOMPUTE if force is None else force))
    if not args.force and cached_cumulant_data_is_current(args):
        print("[cache] existing cumulant summary/data match current settings; reusing them.")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[compute] regenerating cumulant data because cache is absent, forced, or stale.")
    plot_cumulant_heatmaps(args, OUTDIR)


def load_cumulant_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    configure_matplotlib_cache()
    ensure_cumulant_data(force=None)
    with np.load(DATA_PATH) as data:
        nus = np.asarray(data["nus"], dtype=float)
        lambda3 = np.asarray(data["lambda3_over_H"], dtype=float)
        lambda4 = np.asarray(data["lambda4"], dtype=float)
        s3 = np.asarray(data["S3"], dtype=float)
        s4 = np.asarray(data["S4"], dtype=float)
        qng = np.asarray(data["QNG"], dtype=float)
    return nus, lambda3, lambda4, s3, s4, qng


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
