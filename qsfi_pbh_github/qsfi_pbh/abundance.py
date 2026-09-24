#!/usr/bin/env python3
r"""Utility functions and optional standalone abundance plots.

The public functions in this file are imported by ``plot_fig3_abundance.py``.
The file can also be run directly in PyCharm to generate two separate four-panel
heatmaps from the current cumulant summary.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR = PROJECT_DIR / "figures"
SUMMARY_PATH = OUTPUT_DIR / "qsfi_pbh_cumulant_heatmaps_summary.json"

from . import backend as fig2_common
from .collapse import (
    DELTA_C,
    GAMMA_COLLAPSE,
    QNG_CUT_DEFINITION,
    QNG_DEFINITION,
    QNG_MAX,
    VALID_QNG_FIELD,
    qng_from_cumulants,
    qng_is_valid,
)

FIXED_POWER_AREA = fig2_common.POWER_AREA
TARGET_BETA = 1.0e-15
DPI = 300


def ensure_cumulant_summary() -> Dict[str, object]:
    """Load the cumulant summary, generating it first if necessary."""
    if not SUMMARY_PATH.exists():
        fig2_common.ensure_cumulant_data(force=None)
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_power_area_and_response(power_area: float, variance_response: float) -> None:
    power_values = np.asarray(power_area, dtype=float)
    if np.any(~np.isfinite(power_values)) or np.any(power_values <= 0.0):
        raise ValueError("integrated lognormal power area must be positive and finite.")
    if not math.isfinite(variance_response) or variance_response <= 0.0:
        raise ValueError("variance_response must be positive and finite.")


def beta_gaussian(
    power_area: float,
    *,
    variance_response: float,
    delta_c: float = DELTA_C,
    gamma: float = GAMMA_COLLAPSE,
) -> float:
    return math.exp(
        log_beta_gaussian(
            power_area,
            variance_response=variance_response,
            delta_c=delta_c,
            gamma=gamma,
        )
    )


def log_beta_gaussian(
    power_area: float,
    *,
    variance_response: float,
    delta_c: float = DELTA_C,
    gamma: float = GAMMA_COLLAPSE,
) -> float:
    _validate_power_area_and_response(power_area, variance_response)
    sigma = math.sqrt(power_area * variance_response)
    nu_th = delta_c / sigma
    return math.log(gamma) - 0.5 * math.log(2.0 * math.pi) - math.log(nu_th) - 0.5 * nu_th ** 2


def bisect_for_target_beta(
    target_beta: float,
    *,
    variance_response: float,
    delta_c: float = DELTA_C,
    gamma: float = GAMMA_COLLAPSE,
) -> float:
    if not math.isfinite(target_beta) or not (0.0 < target_beta < 1.0):
        raise ValueError("target_beta must satisfy 0 < target_beta < 1.")
    _validate_power_area_and_response(1.0, variance_response)
    lo = 1.0e-8
    hi = 1.0
    for _ in range(240):
        mid = math.sqrt(lo * hi)
        if beta_gaussian(
            mid,
            variance_response=variance_response,
            delta_c=delta_c,
            gamma=gamma,
        ) < target_beta:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def fixed_reduced_cumulants(
    coeff: Dict[str, float],
    lambda3_0: np.ndarray,
    lambda4_0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return S3, S4 and Q_NG fixed by reference-coupling labels."""

    l3_grid, l4_grid = np.meshgrid(lambda3_0, lambda4_0)
    s3 = coeff["S3_per_lambda3_over_H"] * l3_grid
    s4 = (
        coeff["S4_per_lambda4_contact"] * l4_grid
        + coeff["S4_per_lambda3_over_H_squared_exchange"] * l3_grid**2
    )
    qng = qng_from_cumulants(s3, s4)
    return s3, s4, qng


def log_beta_ratio_from_cumulants(
    power_area: float,
    s3: np.ndarray,
    s4: np.ndarray,
    *,
    variance_response: float,
) -> np.ndarray:
    _validate_power_area_and_response(power_area, variance_response)
    nu2 = DELTA_C ** 2 / (power_area * variance_response)
    eng = s3 / 6.0 + DELTA_C * s4 / 24.0
    return nu2 * DELTA_C * eng


def log_beta_ng_from_cumulants(
    power_area: float,
    s3: float,
    s4: float,
    *,
    variance_response: float,
) -> float:
    log_ratio = float(
        log_beta_ratio_from_cumulants(
            power_area,
            np.asarray(s3),
            np.asarray(s4),
            variance_response=variance_response,
        )
    )
    return log_beta_gaussian(power_area, variance_response=variance_response) + log_ratio


def solve_required_power_area(
    coeff: Dict[str, float],
    lambda3_0: float,
    lambda4_0: float,
    target_beta: float,
    variance_response: float,
) -> float:
    """Solve beta_NG(A)=target with S3 and S4 fixed during inversion."""
    gaussian_area = bisect_for_target_beta(target_beta, variance_response=variance_response)
    log_target = math.log(target_beta)
    fixed_s3, fixed_s4, _ = fixed_reduced_cumulants(
        coeff,
        np.asarray([lambda3_0], dtype=float),
        np.asarray([lambda4_0], dtype=float),
    )
    s3_value = float(fixed_s3[0, 0])
    s4_value = float(fixed_s4[0, 0])

    def f(power_area: float) -> float:
        return log_beta_ng_from_cumulants(
            power_area,
            s3_value,
            s4_value,
            variance_response=variance_response,
        ) - log_target

    f_center = f(gaussian_area)
    if f_center >= 0.0:
        hi = gaussian_area
        lo = gaussian_area
        for _ in range(120):
            lo *= 0.8
            if lo <= 1.0e-8:
                break
            if f(lo) <= 0.0:
                break
        if f(lo) > 0.0:
            raise RuntimeError("Could not bracket the target beta below the Gaussian solution.")
    else:
        lo = gaussian_area
        hi = gaussian_area
        for _ in range(120):
            hi *= 1.25
            if hi >= 1.0:
                hi = 1.0
                break
            if f(hi) >= 0.0:
                break
        if f(hi) < 0.0:
            raise RuntimeError("Could not bracket the target beta above the Gaussian solution.")

    for _ in range(220):
        mid = math.sqrt(lo * hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


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


def draw_four_panel_heatmap(
    values: np.ndarray,
    lambda3: np.ndarray,
    lambda4: np.ndarray,
    nus: Iterable[float],
    filename: str,
    colorbar_label: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6), sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.86")
    xgrid, ygrid = np.meshgrid(lambda3, lambda4)

    for idx, (ax, nt) in enumerate(zip(axes.ravel(), nus)):
        panel = np.asarray(values[idx], dtype=float)
        mesh = ax.pcolormesh(xgrid, ygrid, np.ma.masked_invalid(panel), shading="auto", cmap=cmap, norm=heatmap_norm(panel))
        if np.isfinite(panel).any() and np.nanmin(panel) < 0.0 < np.nanmax(panel):
            ax.contour(xgrid, ygrid, panel, levels=[0.0], colors="0.15", linewidths=0.6)
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
        cbar.set_label(colorbar_label, labelpad=2)
        cbar.ax.tick_params(labelsize=7, direction="in", length=2.5, width=0.6)
        cbar.update_ticks()

    for ax in axes.ravel()[len(nus):]:
        ax.set_visible(False)

    fig.tight_layout(w_pad=0.35, h_pad=0.55)
    base = OUTPUT_DIR / filename
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"[saved] {base.with_suffix('.pdf')}")
    print(f"[saved] {base.with_suffix('.png')}")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "mplconfig"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = ensure_cumulant_summary()
    metadata = summary["metadata"]
    coeffs = summary["coefficients"]
    reference_area = float(metadata["power_area"])
    variance_response = float(metadata["variance_response"])
    lambda3 = np.linspace(metadata["lambda3_over_H_range"][0], metadata["lambda3_over_H_range"][1], int(metadata["nscan"]))
    lambda4 = np.linspace(metadata["lambda4_range"][0], metadata["lambda4_range"][1], int(metadata["nscan"]))
    nus = [float(key) for key in coeffs.keys()]
    gaussian_area = bisect_for_target_beta(TARGET_BETA, variance_response=variance_response)

    log10_ratio_panels = []
    required_ratio_panels = []
    abundance_summary: Dict[str, object] = {
        "metadata": {
            "fixed_power_area": FIXED_POWER_AREA,
            "target_beta": TARGET_BETA,
            "delta_c": DELTA_C,
            "gamma": GAMMA_COLLAPSE,
            "qng_definition": QNG_DEFINITION,
            "qng_cut_definition": QNG_CUT_DEFINITION,
            "qng_max": QNG_MAX,
            "variance_response": variance_response,
            "C2_variance_response": variance_response,
            "variance_relation": "sigma2 = A_zeta * C2",
            "fixed_power_sigma2": FIXED_POWER_AREA * variance_response,
            "gaussian_beta_at_fixed_power_area": beta_gaussian(FIXED_POWER_AREA, variance_response=variance_response),
            "gaussian_required_power_area_for_target_beta": gaussian_area,
            "gaussian_required_sigma2_for_target_beta": (
                gaussian_area * variance_response
            ),
        },
        "per_nu": {},
    }

    for nt in nus:
        rec = coeffs[str(nt)]
        s3, s4, qng = fixed_reduced_cumulants(rec, lambda3, lambda4)
        log_ratio = log_beta_ratio_from_cumulants(FIXED_POWER_AREA, s3, s4, variance_response=variance_response)
        log10_ratio = np.where(
            qng_is_valid(qng),
            log_ratio / math.log(10.0),
            np.nan,
        )
        log10_ratio_panels.append(log10_ratio)

        ratios = np.full((lambda4.size, lambda3.size), np.nan, dtype=float)
        excluded_qng = 0
        for iy, l4 in enumerate(lambda4):
            for ix, l3 in enumerate(lambda3):
                try:
                    required_area = solve_required_power_area(
                        rec,
                        float(l3),
                        float(l4),
                        TARGET_BETA,
                        variance_response,
                    )
                except RuntimeError:
                    continue
                if float(qng[iy, ix]) > QNG_MAX:
                    excluded_qng += 1
                    continue
                ratios[iy, ix] = required_area / gaussian_area
        required_ratio_panels.append(ratios)

        abundance_summary["per_nu"][str(nt)] = {
            "log10_beta_ratio_fixed_power_area_min": float(np.nanmin(log10_ratio)) if np.isfinite(log10_ratio).any() else None,
            "log10_beta_ratio_fixed_power_area_max": float(np.nanmax(log10_ratio)) if np.isfinite(log10_ratio).any() else None,
            "Pzeta_NG_over_Pzeta_G_min": float(np.nanmin(ratios)) if np.isfinite(ratios).any() else None,
            "Pzeta_NG_over_Pzeta_G_max": float(np.nanmax(ratios)) if np.isfinite(ratios).any() else None,
            "required_power_qng_excluded_count": int(excluded_qng),
            "QNG_fixed_power_area_max_abs": float(np.nanmax(np.abs(qng))),
        }

    log10_ratio_arr = np.asarray(log10_ratio_panels, dtype=float)
    ratio_arr = np.asarray(required_ratio_panels, dtype=float)
    draw_four_panel_heatmap(log10_ratio_arr, lambda3, lambda4, nus, "qsfi_pbh_beta_ratio_fixed_power", r"$\log_{10}(\beta_{\rm NG}/\beta_{\rm G})$")
    draw_four_panel_heatmap(ratio_arr, lambda3, lambda4, nus, "qsfi_pbh_P_NG_over_P_G_fixed_beta", r"${\cal P}_{\zeta,\rm NG}(k_*)/{\cal P}_{\zeta,\rm G}(k_*)$")

    np.savez_compressed(
        OUTPUT_DIR / "qsfi_pbh_abundance_diagnostics_data.npz",
        nus=np.asarray(nus, dtype=float),
        lambda3_over_H=lambda3,
        lambda4=lambda4,
        log10_beta_ratio_fixed_power_area=log10_ratio_arr,
        Pzeta_NG_over_Pzeta_G=ratio_arr,
    )
    with open(OUTPUT_DIR / "qsfi_pbh_abundance_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(abundance_summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"[saved] {OUTPUT_DIR / 'qsfi_pbh_abundance_diagnostics_summary.json'}")


if __name__ == "__main__":
    main()
