"""Shared publication styling and export helpers."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Iterable

import matplotlib as mpl


# Palette A: charcoal + Oxford navy + muted burgundy.  Curves encode the
# sound-speed ratio by colour; mass is normally placed on the horizontal axis.
CHARCOAL = "#202124"
OXFORD_NAVY = "#344E6C"
BURGUNDY = "#8C3B4A"
WARM_IVORY = "#F4F0E8"
NU_COLORS = (CHARCOAL, OXFORD_NAVY, "#6E6067", BURGUNDY)
R_COLORS = (CHARCOAL, OXFORD_NAVY, BURGUNDY)
POSITIVE_COLOR = BURGUNDY
NEGATIVE_COLOR = OXFORD_NAVY
TEAL_DARK = OXFORD_NAVY  # Backward-compatible semantic alias.
TEAL_LIGHT = "#E9EDF1"
NEUTRAL_DARK = CHARCOAL
NEUTRAL_MID = "#77797C"
NEUTRAL_LIGHT = "#E4E2DE"
INVALID_COLOR = "#D9D9D7"
SEQUENTIAL_COLORS = (
    WARM_IVORY,
    "#D8CCBA",
    "#A99786",
    "#687184",
    OXFORD_NAVY,
)


def apply_publication_style() -> None:
    """Apply the manuscript-wide matplotlib style contract."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "STIX Two Text",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.grid": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "hatch.color": NEUTRAL_MID,
            "hatch.linewidth": 0.35,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sequential_colormap(name: str = "oxford_sequential") -> object:
    """Return the restrained single-direction colormap used for PBH fields."""

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name, SEQUENTIAL_COLORS)
    cmap.set_bad(INVALID_COLOR)
    return cmap


def save_publication_figure(
    figure: object,
    output_stem: Path | str,
    *,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> tuple[Path, ...]:
    """Save a figure as vector PDF/SVG plus a dense PNG preview."""

    stem = Path(output_stem)
    if stem.suffix:
        requested = (stem.suffix.lstrip("."),)
        stem = stem.with_suffix("")
    else:
        requested = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    if not requested or any(fmt not in {"pdf", "svg", "png"} for fmt in requested):
        raise ValueError("formats must be drawn from pdf, svg, and png.")
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in requested:
        destination = stem.with_suffix(f".{fmt}")
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            "dpi": int(dpi),
            "format": fmt,
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                # A dot-prefixed temporary file can retain Finder's hidden flag
                # after an atomic replace on an iCloud-backed Desktop folder.
                prefix=f"tmp-{destination.name}.",
                suffix=f".{fmt}",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            figure.savefig(temporary, **kwargs)
            if temporary.stat().st_size <= 0:
                raise RuntimeError(
                    f"Figure export produced an empty {fmt.upper()} file."
                )
            temporary.replace(destination)
            if hasattr(os, "chflags") and hasattr(stat, "UF_HIDDEN"):
                flags = destination.stat().st_flags
                if flags & stat.UF_HIDDEN:
                    os.chflags(destination, flags & ~stat.UF_HIDDEN)
            outputs.append(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return tuple(outputs)
