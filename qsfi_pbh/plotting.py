"""Line-first JHEP/PRD manuscript figures from production datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .collapse import VALID_QNG_FIELD

from .style import (
    BURGUNDY,
    INVALID_COLOR,
    NEUTRAL_MID,
    NU_COLORS,
    OXFORD_NAVY,
    R_COLORS,
    WARM_IVORY,
    apply_publication_style,
    save_publication_figure,
    sequential_colormap,
)


EXPECTED_RATIOS = np.asarray((0.3, 1.0, 3.0), dtype=float)
EXPECTED_NUS = np.asarray((0.1, 0.3, 0.5, 1.0), dtype=float)
R_MARKERS = ("o", "s", "^")
R_LINESTYLES = ("-", "--", "-.")

_COEFFICIENT_FIELDS = {
    "s3": (
        "C3_hat",
        "C3_hat_standard_error",
    ),
    "contact": (
        "C4_contact_hat",
        "C4_contact_hat_standard_error",
    ),
    "exchange": (
        "C4_exchange_hat",
        "C4_exchange_hat_standard_error",
    ),
}


def _validate_benchmark_grid(ratios: np.ndarray, nus: np.ndarray) -> None:
    if ratios.shape != EXPECTED_RATIOS.shape or not np.allclose(
        ratios,
        EXPECTED_RATIOS,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Manuscript products require freshly generated r = 0.3, 1, 3 data."
        )
    if nus.shape != EXPECTED_NUS.shape or not np.allclose(
        nus,
        EXPECTED_NUS,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Manuscript products require nu_tilde = 0.1, 0.3, 0.5, 1.0."
        )


def collect_coefficient_arrays(summary: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return aligned sound-speed, mass, coefficient, and error arrays."""

    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("summary must contain a non-empty case list.")
    first_nus = np.asarray(cases[0]["nu_tilde"], dtype=float)
    ratios = []
    values = {name: [] for name in _COEFFICIENT_FIELDS}
    errors = {name: [] for name in _COEFFICIENT_FIELDS}
    for case in cases:
        nus = np.asarray(case["nu_tilde"], dtype=float)
        if not np.array_equal(nus, first_nus):
            raise ValueError("All sound-speed cases must use the same masses.")
        ratios.append(float(case["sound_speed"]["ratio"]))
        per_nu = case["per_nu"]
        case_values = {name: [] for name in _COEFFICIENT_FIELDS}
        case_errors = {name: [] for name in _COEFFICIENT_FIELDS}
        for nu in nus:
            coefficients = per_nu[str(float(nu))]["coefficients"]
            for name, (value_key, error_key) in _COEFFICIENT_FIELDS.items():
                case_values[name].append(float(coefficients[value_key]))
                case_errors[name].append(float(coefficients[error_key]))
        for name in _COEFFICIENT_FIELDS:
            values[name].append(case_values[name])
            errors[name].append(case_errors[name])
    ratio_array = np.asarray(ratios, dtype=float)
    _validate_benchmark_grid(ratio_array, first_nus)
    result = {"ratios": ratio_array, "nus": first_nus}
    for name in _COEFFICIENT_FIELDS:
        result[name] = np.asarray(values[name], dtype=float)
        result[f"{name}_error"] = np.asarray(errors[name], dtype=float)
    return result


def r3_coupling_grid_values(
    bundle: Mapping[str, np.ndarray],
    *,
    quantity: str,
) -> np.ndarray:
    """Return the four r=3 coupling-grid panels for a manuscript quantity."""

    ratios = np.asarray(bundle["ratios"], dtype=float)
    nus = np.asarray(bundle["nus"], dtype=float)
    _validate_benchmark_grid(ratios, nus)
    matches = np.flatnonzero(np.isclose(ratios, 3.0, rtol=0.0, atol=1.0e-12))
    if matches.size != 1:
        raise ValueError("The manuscript bundle must contain exactly one r=3 case.")
    ratio_index = int(matches[0])

    if quantity == "log10_beta_ratio":
        panels = np.asarray(bundle["log10_beta_ratio"], dtype=float)[ratio_index]
    elif quantity == "peak_height_ratio":
        panels = np.asarray(bundle["Pzeta_NG_over_Pzeta_G"], dtype=float)[
            ratio_index
        ]
    else:
        raise ValueError(f"Unsupported r=3 grid quantity: {quantity!r}.")

    expected = (nus.size, np.asarray(bundle["lambda4"]).size, np.asarray(bundle["lambda3"]).size)
    if panels.shape != expected:
        raise ValueError(f"The r=3 coupling-grid panels must have shape {expected}.")
    return panels


def sound_speed_coupling_grid_values(
    bundle: Mapping[str, np.ndarray],
    *,
    quantity: str,
) -> np.ndarray:
    """Return all three sound-speed rows for a coupling-grid quantity."""

    ratios = np.asarray(bundle["ratios"], dtype=float)
    nus = np.asarray(bundle["nus"], dtype=float)
    _validate_benchmark_grid(ratios, nus)
    if quantity == "log10_beta_ratio":
        panels = np.asarray(bundle["log10_beta_ratio"], dtype=float)
    elif quantity == "peak_height_ratio":
        panels = np.asarray(bundle["Pzeta_NG_over_Pzeta_G"], dtype=float)
    else:
        raise ValueError(f"Unsupported sound-speed grid quantity: {quantity!r}.")

    expected = (
        ratios.size,
        nus.size,
        np.asarray(bundle["lambda4"]).size,
        np.asarray(bundle["lambda3"]).size,
    )
    if panels.shape != expected:
        raise ValueError(
            f"The sound-speed coupling-grid panels must have shape {expected}."
        )
    return panels


def _reference_aware_colorbar_ticks(
    automatic_ticks: np.ndarray,
    *,
    lower: float,
    reference: float,
    upper: float,
) -> np.ndarray:
    """Keep visible ticks and guarantee labels on both sides of reference."""

    ticks = np.asarray(automatic_ticks, dtype=float)
    visible = ticks[
        np.isfinite(ticks)
        & (ticks >= float(lower))
        & (ticks <= float(upper))
    ]
    if not float(lower) < float(reference) < float(upper):
        return np.unique(visible)
    visible = np.append(visible, float(reference))
    if not np.any(visible < float(reference)):
        visible = np.append(
            visible,
            (float(lower), 0.5 * (float(lower) + float(reference))),
        )
    if not np.any(visible > float(reference)):
        visible = np.append(
            visible,
            (0.5 * (float(reference) + float(upper)), float(upper)),
        )
    return np.unique(visible)


def _format_colorbar_tick_labels(ticks: np.ndarray) -> list[str]:
    """Format one colorbar's ticks without collapsing nearby values."""

    values = np.asarray(ticks, dtype=float)
    regular = values[(values == 0.0) | (np.abs(values) >= 1.0e-3)]
    unique_regular = np.unique(regular)
    gaps = np.diff(unique_regular)
    positive_gaps = gaps[gaps > 0.0]
    decimals = 0
    if positive_gaps.size:
        tolerance = 0.02 * float(np.min(positive_gaps))
        for candidate in range(7):
            if np.all(
                np.abs(regular - np.round(regular, candidate)) <= tolerance
            ):
                decimals = candidate
                break

    labels = []
    for value in values:
        if value != 0.0 and abs(value) < 1.0e-3:
            label = f"{value:.2g}"
        else:
            label = f"{value:.{decimals}f}"
            if "." in label:
                label = label.rstrip("0").rstrip(".")
            if label == "-0":
                label = "0"
        labels.append(label)
    return labels


def _plot_extremum_marker(axis: object, x: float, y: float) -> object:
    """Mark a retained extremum without clipping boundary points."""

    return axis.plot(
        x,
        y,
        marker="*",
        markersize=7.0,
        markerfacecolor="white",
        markeredgecolor="#202124",
        markeredgewidth=0.75,
        linestyle="none",
        clip_on=False,
        zorder=5,
    )[0]


def draw_sound_speed_coupling_grid(
    bundle: Mapping[str, np.ndarray],
    output_path: Path | str,
    *,
    quantity: str,
    include_trend: bool = True,
) -> tuple[Path, ...]:
    """Draw the full 3-by-4 sound-speed comparison in manuscript style.

    Set ``include_trend`` to false when the extrema are summarized separately
    and the coupling-plane panels should occupy the full figure height.
    """

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.gridspec import GridSpec

    apply_publication_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    ratios = np.asarray(bundle["ratios"], dtype=float)
    nus = np.asarray(bundle["nus"], dtype=float)
    lambda3 = np.asarray(bundle["lambda3"], dtype=float)
    lambda4 = np.asarray(bundle["lambda4"], dtype=float)
    panels = sound_speed_coupling_grid_values(bundle, quantity=quantity)

    cmap = LinearSegmentedColormap.from_list(
        f"sound_speed_{quantity}",
        (OXFORD_NAVY, WARM_IVORY, BURGUNDY),
    )
    cmap.set_bad(INVALID_COLOR)
    if quantity == "log10_beta_ratio":
        reference_level = 0.0
        x_label = r"$\lambda_3/H$"
        y_label = r"$\lambda_4$"
        colorbar_label = r"$\log_{10}(\beta_{\rm NG}/\beta_{\rm G})$"
        trend_values = np.asarray(bundle["fixed_power_maxima"], dtype=float)
        trend_ylabel = (
            r"$\max_{\lambda_3,\lambda_4}"
            r"\log_{10}(\beta_{\rm NG}/\beta_{\rm G})$"
        )
        extremum = np.nanargmax
        x_label = r"$\lambda_3/H$"
        y_label = r"$\lambda_4$"
    elif quantity == "peak_height_ratio":
        reference_level = 1.0
        colorbar_label = (
            r"$\mathcal{P}_{\zeta,\rm NG}(k_*)/"
            r"\mathcal{P}_{\zeta,\rm G}(k_*)$"
        )
        trend_values = np.asarray(bundle["fixed_abundance_minima"], dtype=float)
        trend_ylabel = (
            r"$\min_{\lambda_3,\lambda_4}"
            r"[\mathcal{P}_{\zeta,\rm NG}(k_*)/"
            r"\mathcal{P}_{\zeta,\rm G}(k_*)]$"
        )
        extremum = np.nanargmin
        x_label = r"$\lambda_3/H$"
        y_label = r"$\lambda_4$"
    else:
        raise ValueError(f"Unsupported sound-speed grid quantity: {quantity!r}.")

    def make_norm(row: np.ndarray) -> object:
        row_finite = row[np.isfinite(row)]
        if row_finite.size == 0:
            raise ValueError("Every sound-speed row must contain finite values.")
        row_lower = float(np.min(row_finite))
        row_upper = float(np.max(row_finite))
        if np.isclose(row_lower, row_upper):
            padding = max(abs(row_lower) * 0.05, 1.0e-12)
            row_lower -= padding
            row_upper += padding
        if row_lower < reference_level < row_upper:
            return TwoSlopeNorm(
                vmin=row_lower,
                vcenter=reference_level,
                vmax=row_upper,
            )
        return Normalize(vmin=row_lower, vmax=row_upper)

    row_norms = [make_norm(row) for row in panels]

    xgrid, ygrid = np.meshgrid(lambda3, lambda4)
    if include_trend:
        figure = plt.figure(figsize=(7.15, 6.65))
        grid = GridSpec(
            4,
            5,
            figure=figure,
            height_ratios=(1.0, 1.0, 1.0, 0.82),
            width_ratios=(1.0, 1.0, 1.0, 1.0, 0.055),
        )
    else:
        figure = plt.figure(figsize=(7.15, 5.15))
        grid = GridSpec(
            3,
            5,
            figure=figure,
            width_ratios=(1.0, 1.0, 1.0, 1.0, 0.055),
        )
    row_meshes = []
    shared_axis = None
    for ratio_index, ratio in enumerate(ratios):
        row_mesh = None
        for nu_index, nu in enumerate(nus):
            axis = figure.add_subplot(
                grid[ratio_index, nu_index],
                sharex=shared_axis,
                sharey=shared_axis,
            )
            if shared_axis is None:
                shared_axis = axis
            panel = panels[ratio_index, nu_index]
            row_mesh = axis.pcolormesh(
                xgrid,
                ygrid,
                np.ma.masked_invalid(panel),
                shading="auto",
                cmap=cmap,
                norm=row_norms[ratio_index],
                rasterized=True,
            )
            excluded = ~np.isfinite(panel)
            if np.any(excluded):
                axis.contourf(
                    xgrid,
                    ygrid,
                    excluded.astype(float),
                    levels=(0.5, 1.5),
                    colors="none",
                    hatches=("////",),
                    zorder=3,
                )
            panel_finite = panel[np.isfinite(panel)]
            if (
                panel_finite.size
                and float(np.min(panel_finite))
                < reference_level
                < float(np.max(panel_finite))
            ):
                axis.contour(
                    xgrid,
                    ygrid,
                    panel,
                    levels=[reference_level],
                    colors=NEUTRAL_MID,
                    linewidths=0.65,
                    linestyles="--",
                )
            best_flat = int(extremum(panel))
            best_row, best_column = np.unravel_index(best_flat, panel.shape)
            _plot_extremum_marker(
                axis,
                lambda3[best_column],
                lambda4[best_row],
            )
            if ratio_index == 0:
                axis.set_title(rf"$\widetilde{{\nu}}={nu:g}$")
            if ratio_index == ratios.size - 1:
                axis.set_xlabel(x_label)
            else:
                axis.tick_params(labelbottom=False)
            if nu_index == 0:
                axis.set_ylabel(
                    rf"$r_s={ratio:g}$" + "\n" + y_label
                )
            else:
                axis.tick_params(labelleft=False)
        if row_mesh is None:
            raise RuntimeError(
                f"No sound-speed coupling panels were drawn for r_s={ratio:g}."
            )
        row_meshes.append(row_mesh)

    if len(row_meshes) != ratios.size:
        raise RuntimeError("No sound-speed coupling-grid panels were drawn.")
    for ratio_index, row_mesh in enumerate(row_meshes):
        colorbar_axis = figure.add_subplot(grid[ratio_index, 4])
        colorbar = figure.colorbar(row_mesh, cax=colorbar_axis)
        row_norm = row_norms[ratio_index]
        ticks = _reference_aware_colorbar_ticks(
            np.asarray(colorbar.get_ticks(), dtype=float),
            lower=float(row_norm.vmin),
            reference=reference_level,
            upper=float(row_norm.vmax),
        )
        colorbar.set_ticks(
            ticks,
            labels=_format_colorbar_tick_labels(ticks),
        )
        if ratio_index == 1:
            colorbar.set_label(colorbar_label)

    if include_trend:
        trend_axis = figure.add_subplot(grid[3, 0:4])
        nu_markers = ("o", "s", "^", "D")
        nu_linestyles = ("-", "--", "-.", ":")
        for nu_index, (nu, color, marker, linestyle) in enumerate(
            zip(nus, NU_COLORS, nu_markers, nu_linestyles)
        ):
            trend_axis.plot(
                ratios,
                trend_values[:, nu_index],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                label=rf"$\widetilde{{\nu}}={nu:g}$",
            )
        trend_axis.axhline(
            reference_level,
            color=NEUTRAL_MID,
            linewidth=0.65,
            linestyle="--",
            zorder=0,
        )
        trend_axis.set_xscale("log")
        trend_axis.set_xticks(ratios, [f"{ratio:g}" for ratio in ratios])
        trend_axis.set_xlabel(r"$r_s$")
        trend_axis.set_ylabel(trend_ylabel)
        _set_marker_safe_ylim(
            trend_axis,
            trend_values,
            anchors=(reference_level,),
        )
        trend_axis.legend(
            ncol=4,
            loc="best",
            columnspacing=1.1,
            handlelength=2.0,
        )

    figure.subplots_adjust(
        left=0.09,
        right=0.91,
        bottom=0.075 if include_trend else 0.09,
        top=0.965,
        wspace=0.18,
        hspace=0.30 if include_trend else 0.20,
    )
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def draw_r3_coupling_grid(
    bundle: Mapping[str, np.ndarray],
    output_path: Path | str,
    *,
    quantity: str,
) -> tuple[Path, ...]:
    """Draw four r=3 coupling-plane panels with a common color scale."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

    apply_publication_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    nus = np.asarray(bundle["nus"], dtype=float)
    lambda3 = np.asarray(bundle["lambda3"], dtype=float)
    lambda4 = np.asarray(bundle["lambda4"], dtype=float)
    panels = r3_coupling_grid_values(bundle, quantity=quantity)
    finite = panels[np.isfinite(panels)]
    if finite.size == 0:
        raise ValueError(f"No finite values are available for {quantity!r}.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0e-12)
        lower -= padding
        upper += padding

    if quantity == "log10_beta_ratio":
        cmap = LinearSegmentedColormap.from_list(
            "pbh_abundance_shift",
            (OXFORD_NAVY, WARM_IVORY, BURGUNDY),
        )
        cmap.set_bad(INVALID_COLOR)
        if lower < 0.0 < upper:
            norm = TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
        else:
            norm = Normalize(vmin=lower, vmax=upper)
        colorbar_label = r"$\log_{10}(\beta_{\rm NG}/\beta_{\rm G})$"
        extremum = np.nanargmax
        reference_level = 0.0
    elif quantity == "peak_height_ratio":
        cmap = LinearSegmentedColormap.from_list(
            "peak_height_ratio",
            (OXFORD_NAVY, WARM_IVORY, BURGUNDY),
        )
        cmap.set_bad(INVALID_COLOR)
        if lower < 1.0 < upper:
            norm = TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper)
        else:
            norm = Normalize(vmin=lower, vmax=upper)
        colorbar_label = (
            r"$\mathcal{P}_{\zeta,\rm NG}(k_*)/"
            r"\mathcal{P}_{\zeta,\rm G}(k_*)$"
        )
        extremum = np.nanargmin
        reference_level = 1.0
        x_label = r"$\lambda_3/H$"
        y_label = r"$\lambda_4$"
    else:
        raise ValueError(f"Unsupported r=3 grid quantity: {quantity!r}.")

    xgrid, ygrid = np.meshgrid(lambda3, lambda4)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(6.75, 5.15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    mesh = None
    for panel_index, (axis, nu, panel) in enumerate(
        zip(axes.ravel(), nus, panels)
    ):
        masked = np.ma.masked_invalid(panel)
        mesh = axis.pcolormesh(
            xgrid,
            ygrid,
            masked,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        if float(np.nanmin(panel)) < reference_level < float(np.nanmax(panel)):
            axis.contour(
                xgrid,
                ygrid,
                panel,
                levels=[reference_level],
                colors=NEUTRAL_MID,
                linewidths=0.75,
                linestyles="--",
            )
        best_flat = int(extremum(panel))
        best_row, best_column = np.unravel_index(best_flat, panel.shape)
        _plot_extremum_marker(
            axis,
            lambda3[best_column],
            lambda4[best_row],
        )
        axis.set_title(rf"$\widetilde{{\nu}}={nu:g}$")
        if panel_index // 2 == 1:
            axis.set_xlabel(x_label)
        if panel_index % 2 == 0:
            axis.set_ylabel(y_label)

    if mesh is None:
        raise RuntimeError("No r=3 coupling-grid panels were drawn.")
    colorbar = figure.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        fraction=0.045,
        pad=0.035,
        aspect=28,
    )
    colorbar.set_label(colorbar_label)
    figure.subplots_adjust(
        left=0.10,
        right=0.86,
        bottom=0.10,
        top=0.94,
        wspace=0.14,
        hspace=0.24,
    )
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def panel_label_position() -> dict[str, object]:
    """Return the shared below-axis subfigure-label contract."""

    return {"x": 0.5, "y": -0.30, "horizontal_alignment": "center"}


def _add_panel_label(axis: object, label: str) -> None:
    position = panel_label_position()
    axis.text(
        float(position["x"]),
        float(position["y"]),
        f"({label})",
        transform=axis.transAxes,
        ha=str(position["horizontal_alignment"]),
        va="top",
        fontsize=8.0,
        clip_on=False,
    )


def _set_marker_safe_ylim(
    axis: object,
    values: np.ndarray,
    *,
    uncertainties: np.ndarray | None = None,
    anchors: tuple[float, ...] = (),
    padding_fraction: float = 0.065,
) -> tuple[float, float]:
    value_array = np.asarray(values, dtype=float)
    if not np.isfinite(padding_fraction) or padding_fraction <= 0.0:
        raise ValueError("padding_fraction must be positive and finite.")

    if uncertainties is None:
        lower = value_array
        upper = value_array
    else:
        error_array = np.asarray(uncertainties, dtype=float)
        if error_array.shape != value_array.shape:
            raise ValueError("uncertainties must have the same shape as values.")
        if np.any(np.isfinite(error_array) & (error_array < 0.0)):
            raise ValueError("uncertainties must be non-negative.")
        lower = value_array - error_array
        upper = value_array + error_array

    candidates = np.concatenate(
        (
            np.ravel(lower),
            np.ravel(upper),
            np.asarray(anchors, dtype=float).ravel(),
        )
    )
    finite = candidates[np.isfinite(candidates)]
    if finite.size == 0:
        raise ValueError("marker-safe limits require at least one finite value.")
    data_limits = np.asarray((np.min(finite), np.max(finite)), dtype=float)

    transform = axis.yaxis.get_transform()
    transformed = np.asarray(transform.transform(data_limits), dtype=float)
    if not np.all(np.isfinite(transformed)) or transformed[1] <= transformed[0]:
        raise ValueError("marker-safe limits require a non-degenerate y extent.")
    padding = padding_fraction * float(transformed[1] - transformed[0])
    padded = np.asarray(
        transform.inverted().transform(
            (transformed[0] - padding, transformed[1] + padding)
        ),
        dtype=float,
    )
    limits = (float(padded[0]), float(padded[1]))
    axis.set_ylim(*limits)
    return limits


def _plot_ratio_series(
    axis: object,
    ratios: np.ndarray,
    nus: np.ndarray,
    values: np.ndarray,
) -> tuple[object, ...]:
    handles = []
    for index, (ratio, color, marker, linestyle) in enumerate(
        zip(ratios, R_COLORS, R_MARKERS, R_LINESTYLES)
    ):
        (line,) = axis.plot(
            nus,
            values[index],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=rf"$r_s={ratio:g}$",
        )
        handles.append(line)
    axis.set_xticks(nus, [f"{nu:g}" for nu in nus])
    axis.set_xlim(float(nus[0]) - 0.035, float(nus[-1]) + 0.035)
    return tuple(handles)


def _plot_coefficient_series(
    axis: object,
    ratios: np.ndarray,
    nus: np.ndarray,
    coefficients: np.ndarray,
    uncertainties: np.ndarray,
) -> tuple[object, ...]:
    handles = []
    for ratio_index, (ratio, color, marker, linestyle) in enumerate(
        zip(ratios, R_COLORS, R_MARKERS, R_LINESTYLES)
    ):
        handle = axis.errorbar(
            nus,
            coefficients[ratio_index],
            yerr=uncertainties[ratio_index],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            capsize=1.7,
            elinewidth=0.65,
            label=rf"$r_s={ratio:g}$",
        )
        handles.append(handle)
    return tuple(handles)


def draw_coefficient_summary(
    arrays: Mapping[str, np.ndarray],
    output_path: Path | str,
) -> tuple[Path, ...]:
    """Draw absolute hatted projected coefficients versus nu_tilde."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style()
    ratios = np.asarray(arrays["ratios"], dtype=float)
    nus = np.asarray(arrays["nus"], dtype=float)
    _validate_benchmark_grid(ratios, nus)
    expected = (ratios.size, nus.size)
    for name in _COEFFICIENT_FIELDS:
        if np.asarray(arrays[name]).shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        if np.asarray(arrays[f"{name}_error"]).shape != expected:
            raise ValueError(f"{name}_error must have shape {expected}.")

    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.72), sharex=True)
    panels = (
        ("s3", r"$|\widehat{\mathcal{C}}_3|$"),
        ("contact", r"$|\widehat{\mathcal{C}}_{4,c}|$"),
        ("exchange", r"$|\widehat{\mathcal{C}}_{4,e}|$"),
    )
    legend_handles: tuple[object, ...] = ()
    for panel_index, (axis, (name, ylabel)) in enumerate(zip(axes, panels)):
        coefficients = np.abs(np.asarray(arrays[name], dtype=float))
        uncertainties = np.asarray(arrays[f"{name}_error"], dtype=float)
        panel_handles = _plot_coefficient_series(
            axis,
            ratios,
            nus,
            coefficients,
            uncertainties,
        )
        if panel_index == 0:
            legend_handles = panel_handles
        axis.set_xticks(nus, [f"{nu:g}" for nu in nus])
        axis.set_xlim(float(nus[0]) - 0.035, float(nus[-1]) + 0.035)
        axis.set_yscale("log")
        _set_marker_safe_ylim(
            axis,
            coefficients,
            uncertainties=uncertainties,
        )
        axis.set_xlabel(r"$\widetilde\nu$")
        axis.set_ylabel(ylabel)
        _add_panel_label(axis, chr(ord("a") + panel_index))

    figure.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        columnspacing=1.5,
        handlelength=2.2,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.29, top=0.80, wspace=0.38)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def draw_s3_curves(
    arrays: Mapping[str, np.ndarray],
    output_path: Path | str,
) -> tuple[Path, ...]:
    """Draw four reference-amplitude S3 curves for one sound-speed ratio."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style()
    nus = np.asarray(arrays["nu_tilde"], dtype=float)
    lambda3 = np.asarray(arrays["lambda3_over_H"], dtype=float)
    s3 = np.asarray(arrays["S3"], dtype=float)
    errors = np.asarray(arrays["S3_standard_error"], dtype=float)
    expected = (nus.size, lambda3.size)
    if s3.shape != expected or errors.shape != expected:
        raise ValueError(f"S3 curves and errors must have shape {expected}.")
    if nus.size != len(NU_COLORS):
        raise ValueError("The manuscript S3 figure requires four mass curves.")

    figure, axis = plt.subplots(figsize=(4.15, 3.05))
    linestyles = ("-", "--", "-.", ":")
    for nu, values, uncertainty, color, linestyle in zip(
        nus,
        s3,
        errors,
        NU_COLORS,
        linestyles,
    ):
        axis.plot(
            lambda3,
            values,
            color=color,
            linestyle=linestyle,
            label=rf"$\widetilde{{\nu}}={nu:g}$",
        )
        axis.fill_between(
            lambda3,
            values - uncertainty,
            values + uncertainty,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
    axis.axhline(0.0, color=NEUTRAL_MID, linewidth=0.65, linestyle="--")
    axis.set_xlim(float(lambda3[0]), float(lambda3[-1]))
    _set_marker_safe_ylim(axis, s3, uncertainties=errors, anchors=(0.0,))
    axis.set_xlabel(r"$\lambda_3/H$")
    axis.set_ylabel(r"$S_3$")
    axis.legend(ncol=2, loc="best", columnspacing=1.0, handlelength=2.1)
    figure.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.96)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def draw_cumulant_contributions(
    arrays: Mapping[str, np.ndarray],
    output_path: Path | str,
) -> tuple[Path, ...]:
    """Draw bispectrum and trispectrum terms at r=3 fixed-abundance minima."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style()
    nus = np.asarray(arrays["nu_tilde"], dtype=float)
    s3 = np.asarray(arrays["S3"], dtype=float)
    s4_over_8 = np.asarray(arrays["S4_over_8"], dtype=float)
    expected = (EXPECTED_NUS.size,)
    for name, values in (("nu_tilde", nus), ("S3", s3), ("S4_over_8", s4_over_8)):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(nus, EXPECTED_NUS, rtol=0.0, atol=1.0e-12):
        raise ValueError("The contribution figure requires nu_tilde = 0.1, 0.3, 0.5, 1.0.")

    positions = np.arange(nus.size, dtype=float)
    width = 0.34
    figure, axis = plt.subplots(figsize=(4.3, 2.9))
    axis.bar(
        positions - width / 2.0,
        s3,
        width,
        color=BURGUNDY,
        edgecolor="white",
        linewidth=0.45,
        label=r"$S_3$ (bispectrum)",
        zorder=2,
    )
    axis.bar(
        positions + width / 2.0,
        s4_over_8,
        width,
        color=OXFORD_NAVY,
        edgecolor="white",
        linewidth=0.45,
        label=r"$S_4/8$ (trispectrum)",
        zorder=2,
    )
    axis.axhline(0.0, color=NEUTRAL_MID, linewidth=0.7, zorder=1)
    axis.set_xticks(positions, [f"{nu:g}" for nu in nus])
    axis.set_xlim(-0.55, positions[-1] + 0.55)
    lower = float(min(np.min(s3), np.min(s4_over_8), 0.0))
    upper = float(max(np.max(s3), np.max(s4_over_8), 0.0))
    padding = max(0.12 * (upper - lower), 0.05)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel(r"$\widetilde\nu$")
    axis.set_ylabel(r"Contribution to $\mathcal{Q}_{\rm NG}$")
    axis.legend(loc="upper right", ncol=1, handlelength=1.5)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.96)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def draw_relative_contribution_bars(
    arrays: Mapping[str, np.ndarray],
    output_path: Path | str,
) -> tuple[Path, ...]:
    """Draw signed bispectrum and trispectrum contributions."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style()
    nus = np.asarray(arrays["nu_tilde"], dtype=float)
    s3 = np.asarray(arrays["S3"], dtype=float)
    s4_over_8 = np.asarray(arrays["S4_over_8"], dtype=float)

    if nus.shape != EXPECTED_NUS.shape or not np.array_equal(nus, EXPECTED_NUS):
        raise ValueError("The contribution bars require the four manuscript masses.")
    if s3.shape != nus.shape or s4_over_8.shape != nus.shape:
        raise ValueError("Optimum contribution bars must follow the mass axis.")
    if np.any(~np.isfinite(s3)) or np.any(~np.isfinite(s4_over_8)):
        raise ValueError("Optimum contribution bars must be finite.")

    figure, bar_axis = plt.subplots(figsize=(3.7, 3.2))
    positions = np.arange(nus.size, dtype=float)
    width = 0.34
    bar_axis.bar(
        positions - width / 2.0,
        s3,
        width,
        color=BURGUNDY,
        edgecolor="none",
        linewidth=0.0,
        label=r"$S_3$ (bispectrum)",
        zorder=2,
    )
    bar_axis.bar(
        positions + width / 2.0,
        s4_over_8,
        width,
        color=OXFORD_NAVY,
        edgecolor="none",
        linewidth=0.0,
        label=r"$S_4/8$ (trispectrum)",
        zorder=2,
    )
    bar_axis.axhline(0.0, color=NEUTRAL_MID, linewidth=0.7, zorder=1)
    bar_axis.set_xticks(positions, [f"{nu:g}" for nu in nus])
    bar_axis.set_xlabel(r"$\widetilde\nu$")
    bar_axis.legend(loc="upper right")

    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.96)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def draw_fixed_power_summary(
    bundle: Mapping[str, np.ndarray],
    output_path: Path | str,
) -> tuple[Path, ...]:
    """Draw the fixed-peak-height PBH abundance change for all 12 cases."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style()
    ratios = np.asarray(bundle["ratios"], dtype=float)
    nus = np.asarray(bundle["nus"], dtype=float)
    maxima = np.asarray(bundle["fixed_power_maxima"], dtype=float)
    _validate_benchmark_grid(ratios, nus)
    if maxima.shape != (ratios.size, nus.size):
        raise ValueError("fixed_power_maxima has an incompatible shape.")

    figure, axis = plt.subplots(figsize=(3.45, 2.62))
    handles = _plot_ratio_series(axis, ratios, nus, maxima)
    axis.axhline(0.0, color=NEUTRAL_MID, linestyle=":", linewidth=0.75, zorder=0)
    _set_marker_safe_ylim(axis, maxima, anchors=(0.0,))
    axis.set_xlabel(r"$\widetilde\nu$")
    axis.set_ylabel(
        r"$\max_{\lambda_3,\lambda_4}\,\log_{10}"
        r"(\beta_{\rm NG}/\beta_{\rm G})$"
    )
    axis.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="best",
        ncol=1,
        handlelength=2.0,
    )
    figure.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.96)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def draw_fixed_abundance_summary(
    bundle: Mapping[str, np.ndarray],
    output_path: Path | str,
) -> tuple[Path, ...]:
    """Draw the fixed-abundance peak-height ratio for all 12 cases."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style()
    ratios = np.asarray(bundle["ratios"], dtype=float)
    nus = np.asarray(bundle["nus"], dtype=float)
    minima = np.asarray(bundle["fixed_abundance_minima"], dtype=float)
    _validate_benchmark_grid(ratios, nus)
    if minima.shape != (ratios.size, nus.size):
        raise ValueError("fixed_abundance_minima has an incompatible shape.")

    figure, axis = plt.subplots(figsize=(3.45, 2.62))
    handles = _plot_ratio_series(axis, ratios, nus, minima)
    axis.axhline(1.0, color=NEUTRAL_MID, linestyle=":", linewidth=0.75, zorder=0)
    axis.set_xlabel(r"$\widetilde\nu$")
    axis.set_ylabel(
        r"$\min_{\lambda_3,\lambda_4}\,[\mathcal{P}_{\zeta,\rm NG}(k_*)/"
        r"\mathcal{P}_{\zeta,\rm G}(k_*)]$"
    )
    _set_marker_safe_ylim(axis, minima, anchors=(1.0,))
    axis.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="best",
        ncol=1,
        handlelength=2.0,
    )
    figure.subplots_adjust(left=0.22, right=0.97, bottom=0.19, top=0.96)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)


def _best_flat_index(
    score: np.ndarray,
    *,
    excluded: set[int],
    maximize: bool,
) -> int:
    values = np.asarray(score, dtype=float).ravel().copy()
    for index in excluded:
        values[index] = -np.inf if maximize else np.inf
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("No unused finite benchmark remains for regime selection.")
    return int(np.nanargmax(values) if maximize else np.nanargmin(values))


def select_representative_cases(
    bundle: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    """Select four distinct representative regimes from the 3 x 4 grid."""

    ratios = np.asarray(bundle["ratios"], dtype=float)
    nus = np.asarray(bundle["nus"], dtype=float)
    minima = np.asarray(bundle["fixed_abundance_minima"], dtype=float)
    s3 = np.asarray(bundle["s3_at_fixed_abundance_optimum"], dtype=float)
    s4_over_8 = np.asarray(
        bundle["s4_over_8_at_fixed_abundance_optimum"],
        dtype=float,
    )
    expected = (ratios.size, nus.size)
    for name, values in (
        ("fixed_abundance_minima", minima),
        ("s3_at_fixed_abundance_optimum", s3),
        ("s4_over_8_at_fixed_abundance_optimum", s4_over_8),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")

    excluded: set[int] = set()
    weak = _best_flat_index(np.abs(minima - 1.0), excluded=excluded, maximize=False)
    excluded.add(weak)
    maximum_reduction = _best_flat_index(minima, excluded=excluded, maximize=False)
    excluded.add(maximum_reduction)
    dominance = np.abs(s4_over_8) / np.maximum(np.abs(s3), 1.0e-15)
    trispectrum = _best_flat_index(dominance, excluded=excluded, maximize=True)
    excluded.add(trispectrum)
    midpoint = 0.5 * (minima.ravel()[weak] + minima.ravel()[maximum_reduction])
    intermediate = _best_flat_index(
        np.abs(minima - midpoint),
        excluded=excluded,
        maximize=False,
    )

    chosen = (
        ("Weak effect", weak),
        ("Intermediate", intermediate),
        ("Maximum reduction", maximum_reduction),
        ("Trispectrum-dominated", trispectrum),
    )
    rows: list[dict[str, object]] = []
    gaussian_peak_height = float(bundle["gaussian_required_peak_height"])
    for regime, flat_index in chosen:
        ratio_index, nu_index = np.unravel_index(flat_index, expected)
        peak_ratio = float(minima[ratio_index, nu_index])
        s3_value = float(s3[ratio_index, nu_index])
        s4_value = float(s4_over_8[ratio_index, nu_index])
        rows.append(
            {
                "regime": regime,
                "ratio_index": int(ratio_index),
                "nu_index": int(nu_index),
                "r": float(ratios[ratio_index]),
                "nu_tilde": float(nus[nu_index]),
                "lambda3_over_H": float(
                    bundle["lambda3_at_fixed_abundance_optimum"][ratio_index, nu_index]
                ),
                "lambda4": float(
                    bundle["lambda4_at_fixed_abundance_optimum"][ratio_index, nu_index]
                ),
                "s3": s3_value,
                "s4_over_8": s4_value,
                "qng": s3_value + s4_value,
                "peak_height_ratio": peak_ratio,
                "gaussian_peak_height": gaussian_peak_height,
                "optimized_ng_peak_height": gaussian_peak_height * peak_ratio,
                "peak_height_reduction_percent": 100.0 * (1.0 - peak_ratio),
                "trispectrum_dominance": float(dominance[ratio_index, nu_index]),
            }
        )
    return rows


def draw_case_coupling_grid(
    arrays: Mapping[str, np.ndarray],
    output_path: Path | str,
    *,
    value_key: str,
    qng_max: float,
) -> tuple[Path, ...]:
    """Draw one four-mass coupling grid in the established manuscript style."""

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.ticker import ScalarFormatter

    apply_publication_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        }
    )
    nus = np.asarray(arrays["nu_tilde"], dtype=float)
    lambda3 = np.asarray(arrays["lambda3_over_H"], dtype=float)
    lambda4 = np.asarray(arrays["lambda4"], dtype=float)
    values = np.asarray(arrays[value_key], dtype=float)
    qng = np.asarray(arrays["Q_NG"], dtype=float)
    expected = (nus.size, lambda4.size, lambda3.size)
    if values.shape != expected or qng.shape != expected:
        raise ValueError(f"Coupling panels must have shape {expected}.")
    if nus.size != 4:
        raise ValueError("The manuscript coupling layout requires four masses.")

    if value_key == "S4":
        panels = np.where(qng <= qng_max, values, np.nan)
        colorbar_label = r"$S_4$"
        reference_level = 0.0
        per_panel_scale = True
        extremum = np.nanargmax
    elif value_key == "log10_beta_ratio":
        panels = values
        colorbar_label = r"$\log_{10}(\beta_{\rm NG}/\beta_{\rm G})$"
        reference_level = 0.0
        per_panel_scale = False
        extremum = np.nanargmax
    elif value_key == "Pzeta_NG_over_Pzeta_G":
        panels = values
        colorbar_label = (
            r"$\mathcal{P}_{\zeta,\rm NG}(k_*)/"
            r"\mathcal{P}_{\zeta,\rm G}(k_*)$"
        )
        reference_level = 1.0
        per_panel_scale = False
        extremum = np.nanargmin
    else:
        raise ValueError(f"Unsupported coupling field: {value_key!r}.")

    cmap = LinearSegmentedColormap.from_list(
        f"qsfi_{value_key}",
        (OXFORD_NAVY, WARM_IVORY, BURGUNDY),
    )
    cmap.set_bad(INVALID_COLOR)
    finite = panels[np.isfinite(panels)]
    if finite.size == 0:
        raise ValueError("No finite coupling values are available.")
    global_lower = float(np.min(finite))
    global_upper = float(np.max(finite))

    def make_norm(lower: float, upper: float) -> object:
        if np.isclose(lower, upper):
            padding = max(abs(lower) * 0.05, 1.0e-12)
            lower -= padding
            upper += padding
        if lower < reference_level < upper:
            return TwoSlopeNorm(
                vmin=lower,
                vcenter=reference_level,
                vmax=upper,
            )
        return Normalize(vmin=lower, vmax=upper)

    global_norm = make_norm(global_lower, global_upper)
    xgrid, ygrid = np.meshgrid(lambda3, lambda4)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(6.75, 5.15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    shared_mesh = None
    for panel_index, (axis, nu, panel) in enumerate(
        zip(axes.ravel(), nus, panels)
    ):
        panel_finite = panel[np.isfinite(panel)]
        if panel_finite.size == 0:
            raise ValueError(f"No finite values for nu_tilde={nu:g}.")
        norm = (
            make_norm(float(np.min(panel_finite)), float(np.max(panel_finite)))
            if per_panel_scale
            else global_norm
        )
        mesh = axis.pcolormesh(
            xgrid,
            ygrid,
            np.ma.masked_invalid(panel),
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        shared_mesh = mesh
        if (
            float(np.min(panel_finite))
            < reference_level
            < float(np.max(panel_finite))
        ):
            axis.contour(
                xgrid,
                ygrid,
                panel,
                levels=[reference_level],
                colors=NEUTRAL_MID,
                linewidths=0.75,
                linestyles="--",
            )
        best_flat = int(extremum(panel))
        best_row, best_column = np.unravel_index(best_flat, panel.shape)
        _plot_extremum_marker(
            axis,
            lambda3[best_column],
            lambda4[best_row],
        )
        axis.set_title(rf"$\widetilde{{\nu}}={nu:g}$")
        if panel_index // 2 == 1:
            axis.set_xlabel(
                r"$\lambda_3/H$"
            )
        if panel_index % 2 == 0:
            axis.set_ylabel(r"$\lambda_4$")
        if per_panel_scale:
            colorbar = figure.colorbar(
                mesh,
                ax=axis,
                fraction=0.046,
                pad=0.025,
            )
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            colorbar.formatter = formatter
            colorbar.set_label(colorbar_label, labelpad=2)
            colorbar.update_ticks()

    if shared_mesh is None:
        raise RuntimeError("No coupling panels were drawn.")
    if not per_panel_scale:
        colorbar = figure.colorbar(
            shared_mesh,
            ax=axes.ravel().tolist(),
            fraction=0.045,
            pad=0.035,
            aspect=28,
        )
        colorbar.set_label(colorbar_label)
        figure.subplots_adjust(
            left=0.10,
            right=0.86,
            bottom=0.10,
            top=0.94,
            wspace=0.14,
            hspace=0.24,
        )
    else:
        figure.tight_layout(w_pad=0.35, h_pad=0.55)
    try:
        return save_publication_figure(figure, output_path)
    finally:
        plt.close(figure)
