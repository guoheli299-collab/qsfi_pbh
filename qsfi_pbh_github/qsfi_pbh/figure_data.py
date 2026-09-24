"""Stable, safe and independently readable datasets for every final figure."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .config import PRODUCTION_CONTRACT_VERSION


EXPECTED_DATA_PATHS = tuple(
    Path(path)
    for path in (
        "manuscript/hatted_cumulant_coefficients.npz",
        "manuscript/r3_cumulant_contributions.npz",
        "manuscript/sound_speed_log10_beta_ratio_full.npz",
        "manuscript/sound_speed_comparison_full.npz",
    )
)


class FigureDataError(ValueError):
    """Raised when a figure dataset is missing, unsafe or incompatible."""


@dataclass(frozen=True)
class FigureData:
    path: Path
    metadata: dict[str, object]
    arrays: dict[str, np.ndarray]


def _contract_error(
    field: str,
    found: object,
    expected: object,
) -> FigureDataError:
    return FigureDataError(
        f"Production metadata field {field!r} is {found!r}; "
        f"expected {expected!r}. Please recompute with this release."
    )


def _validate_float(field: str, found: object, expected: float) -> None:
    try:
        value = float(found)
    except (TypeError, ValueError) as error:
        raise _contract_error(field, found, expected) from error
    if not np.isclose(value, expected, rtol=0.0, atol=1.0e-15):
        raise _contract_error(field, found, expected)


def _validate_range(
    field: str,
    found: object,
    expected: tuple[float, float],
) -> None:
    if not isinstance(found, (list, tuple)) or len(found) != 2:
        raise _contract_error(field, found, list(expected))
    for index, expected_value in enumerate(expected):
        _validate_float(f"{field}[{index}]", found[index], expected_value)


def validate_production_metadata(metadata: Mapping[str, object]) -> None:
    """Require the exact manuscript production-data contract."""

    if not isinstance(metadata, Mapping):
        raise FigureDataError(
            "Production metadata must be a mapping. "
            "Please recompute with this release."
        )
    exact_fields = {
        "production_contract_version": PRODUCTION_CONTRACT_VERSION,
        "mode": "production",
        "manuscript_usable": True,
        "lambda3_points": 2001,
        "lambda4_points": 2001,
    }
    for field, expected in exact_fields.items():
        found = metadata.get(field)
        if found != expected or (
            field == "manuscript_usable" and found is not True
        ):
            raise _contract_error(field, found, expected)
    _validate_range("lambda3_range", metadata.get("lambda3_range"), (-1.0, 1.0))
    _validate_range("lambda4_range", metadata.get("lambda4_range"), (-1.0, 1.0))
    _validate_float("lambda2_over_H", metadata.get("lambda2_over_H"), 0.38)
    _validate_float("lambda3_step", metadata.get("lambda3_step"), 0.001)
    _validate_float("lambda4_step", metadata.get("lambda4_step"), 0.001)


def _safe_array(name: str, value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise FigureDataError(f"{name!r} must not use object dtype.")
    return array


def write_figure_data(
    path: Path | str,
    *,
    metadata: Mapping[str, object],
    arrays: Mapping[str, object],
) -> Path:
    """Atomically write one pickle-free compressed figure dataset."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "kind" not in metadata:
        raise FigureDataError("metadata must define the figure kind.")
    validate_production_metadata(metadata)
    payload = {
        str(name): _safe_array(str(name), value)
        for name, value in arrays.items()
    }
    if "metadata_json" in payload:
        raise FigureDataError("metadata_json is a reserved field name.")
    payload["metadata_json"] = np.asarray(
        json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _strictly_increasing(name: str, values: np.ndarray) -> None:
    if values.ndim != 1 or values.size < 2:
        raise FigureDataError(f"{name!r} must be a one-dimensional axis.")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
        raise FigureDataError(f"{name!r} must be finite and strictly increasing.")


def load_figure_data(
    path: Path | str,
    *,
    expected_kind: str,
) -> FigureData:
    """Load and validate one dataset without enabling pickle."""

    source = Path(path)
    if not source.is_file():
        raise FigureDataError(f"Missing figure dataset: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise FigureDataError(f"{source} has no metadata_json field.")
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise FigureDataError(f"{source} has invalid metadata JSON.") from error
        arrays = {
            name: _safe_array(name, archive[name])
            for name in archive.files
            if name != "metadata_json"
        }

    if not isinstance(metadata, dict):
        raise FigureDataError("metadata_json must decode to a mapping.")
    if metadata.get("kind") != expected_kind:
        raise FigureDataError(
            f"Expected figure kind {expected_kind!r}, "
            f"found {metadata.get('kind')!r}."
        )
    validate_production_metadata(metadata)
    for axis_name in (
        "lambda3_over_H",
        "lambda4",
        "nu_tilde",
    ):
        if axis_name in arrays:
            _strictly_increasing(axis_name, arrays[axis_name])
    return FigureData(path=source, metadata=metadata, arrays=arrays)
