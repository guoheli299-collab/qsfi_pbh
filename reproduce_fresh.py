#!/usr/bin/env python3
"""Run strict production science from a newly created empty cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from qsfi_pbh.figure_data import validate_production_metadata
from qsfi_pbh.reporting import REPORT_RELATIVE_PATH


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FRESH_ROOT = PACKAGE_DIR / "fresh_runs"


@dataclass(frozen=True)
class FreshRunPaths:
    """All scientific and presentation paths owned by one fresh run."""

    run_dir: Path
    cache_root: Path
    output_root: Path
    figure_root: Path
    mplconfig_root: Path
    manifest_path: Path

    @classmethod
    def for_run_dir(cls, run_dir: Path | str) -> "FreshRunPaths":
        root = Path(run_dir).expanduser().resolve()
        return cls(
            run_dir=root,
            cache_root=root / "cache",
            output_root=root / "figure_data",
            figure_root=root / "figures",
            mplconfig_root=root / "mplconfig",
            manifest_path=root / "fresh_run_manifest.json",
        )


def utc_timestamp() -> str:
    """Return a stable UTC timestamp for manifests and directory names."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_dir(
    *,
    fresh_root: Path | str = DEFAULT_FRESH_ROOT,
    timestamp: str | None = None,
) -> Path:
    """Return a new timestamp-named production directory candidate."""

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(fresh_root).expanduser().resolve() / f"production_{timestamp}"


def prepare_fresh_run(run_dir: Path | str) -> FreshRunPaths:
    """Exclusively create a run and its initially empty scientific roots."""

    paths = FreshRunPaths.for_run_dir(run_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    for directory in (
        paths.cache_root,
        paths.output_root,
        paths.figure_root,
        paths.mplconfig_root,
    ):
        directory.mkdir()
    return paths


def build_compute_command(
    *,
    package_dir: Path | str,
    paths: FreshRunPaths,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the production-only command using isolated absolute roots."""

    package = Path(package_dir).expanduser().resolve()
    return (
        str(python_executable),
        str(package / "compute_results.py"),
        "--cache-root",
        str(paths.cache_root),
        "--output-root",
        str(paths.output_root),
        "--figure-root",
        str(paths.figure_root),
    )


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace the JSON manifest inside the fresh run."""

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
            Path(temporary_name).unlink(missing_ok=True)


def audit_fresh_outputs(
    paths: FreshRunPaths,
    *,
    expected_data_paths: Sequence[Path] | None = None,
    expected_figure_paths: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Require the exact output contract and production metadata."""

    if expected_data_paths is None:
        from qsfi_pbh.figure_data import EXPECTED_DATA_PATHS

        expected_data_paths = EXPECTED_DATA_PATHS
    if expected_figure_paths is None:
        from make_figures import EXPECTED_FIGURE_PATHS

        expected_figure_paths = EXPECTED_FIGURE_PATHS

    expected_data = {Path(path) for path in expected_data_paths}
    expected_figures = {Path(path) for path in expected_figure_paths}
    found_data = {
        path.relative_to(paths.output_root)
        for path in paths.output_root.rglob("*.npz")
    }
    found_figures = {
        path.relative_to(paths.figure_root)
        for path in paths.figure_root.rglob("*")
        if path.is_file() and path.suffix in {".pdf", ".svg", ".png"}
    }
    found_reports = {
        path.relative_to(paths.output_root)
        for path in paths.output_root.rglob("*.txt")
    }
    if found_data != expected_data:
        missing = sorted(str(path) for path in expected_data - found_data)
        extra = sorted(str(path) for path in found_data - expected_data)
        raise RuntimeError(
            f"Fresh data contract mismatch; missing={missing}, extra={extra}."
        )
    if found_figures != expected_figures:
        missing = sorted(str(path) for path in expected_figures - found_figures)
        extra = sorted(str(path) for path in found_figures - expected_figures)
        raise RuntimeError(
            f"Fresh figure contract mismatch; missing={missing}, extra={extra}."
        )
    if found_reports != {REPORT_RELATIVE_PATH}:
        missing = sorted(str(path) for path in {REPORT_RELATIVE_PATH} - found_reports)
        extra = sorted(str(path) for path in found_reports - {REPORT_RELATIVE_PATH})
        raise RuntimeError(
            f"Fresh report contract mismatch; missing={missing}, extra={extra}."
        )

    for relative_path in sorted(expected_data):
        with np.load(paths.output_root / relative_path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise RuntimeError(f"{relative_path} has no metadata_json field.")
            metadata = json.loads(str(archive["metadata_json"].item()))
        validate_production_metadata(metadata)

    empty_figures = [
        str(relative_path)
        for relative_path in sorted(expected_figures)
        if (paths.figure_root / relative_path).stat().st_size <= 0
    ]
    if empty_figures:
        raise RuntimeError(f"Fresh figure files are empty: {empty_figures}.")
    report_path = paths.output_root / REPORT_RELATIVE_PATH
    if report_path.stat().st_size <= 0:
        raise RuntimeError("Fresh manuscript report is empty.")
    report = report_path.read_text(encoding="utf-8")
    required_headings = (
        "[fixed_power_extrema]",
        "[fixed_abundance_extrema]",
        "[projected_coefficients]",
    )
    missing_headings = [heading for heading in required_headings if heading not in report]
    if missing_headings:
        raise RuntimeError(
            f"Fresh manuscript report is missing required heading(s): {missing_headings}."
        )
    return {
        "data_files": len(found_data),
        "report_files": len(found_reports),
        "figure_files": len(found_figures),
        "all_datasets_production": True,
        "all_datasets_manuscript_usable": True,
    }


Runner = Callable[..., object]
Auditor = Callable[[FreshRunPaths], dict[str, object]]


def run_fresh(
    run_dir: Path | str,
    *,
    package_dir: Path | str = PACKAGE_DIR,
    python_executable: str = sys.executable,
    runner: Runner = subprocess.run,
    auditor: Auditor = audit_fresh_outputs,
    environ: Mapping[str, str] | None = None,
) -> FreshRunPaths:
    """Create and execute one non-resumable, isolated production run."""

    package = Path(package_dir).expanduser().resolve()
    paths = prepare_fresh_run(run_dir)
    scientific_roots = (
        paths.cache_root,
        paths.output_root,
        paths.figure_root,
    )
    empty_before_start = all(not any(root.iterdir()) for root in scientific_roots)
    if not empty_before_start:
        raise RuntimeError("Fresh scientific roots were not empty before launch.")
    command = build_compute_command(
        package_dir=package,
        paths=paths,
        python_executable=python_executable,
    )
    child_environment = dict(os.environ if environ is None else environ)
    child_environment.update(
        {
            "MPLCONFIGDIR": str(paths.mplconfig_root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    manifest: dict[str, object] = {
        "status": "running",
        "started_at_utc": utc_timestamp(),
        "completed_at_utc": None,
        "python_executable": str(python_executable),
        "command": list(command),
        "package_dir": str(package),
        "run_dir": str(paths.run_dir),
        "cache_root": str(paths.cache_root),
        "output_root": str(paths.output_root),
        "figure_root": str(paths.figure_root),
        "mplconfig_root": str(paths.mplconfig_root),
        "cache_policy": "fresh_only_no_reuse",
        "fresh_cache_verified_empty_before_start": empty_before_start,
        "existing_project_cache_excluded": str(package / ".cache"),
        "existing_project_data_excluded": str(package / "figure_data"),
        "existing_project_figures_excluded": str(package / "figures"),
    }
    _write_manifest(paths.manifest_path, manifest)
    print(f"[fresh] run directory: {paths.run_dir}", flush=True)
    print(f"[fresh] empty cache: {paths.cache_root}", flush=True)
    print(f"[fresh] command: {' '.join(command)}", flush=True)
    previous_mplconfig = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(paths.mplconfig_root)
    try:
        runner(command, check=True, cwd=package, env=child_environment)
        manifest["audit"] = auditor(paths)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = utc_timestamp()
        _write_manifest(paths.manifest_path, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_at_utc"] = utc_timestamp()
        _write_manifest(paths.manifest_path, manifest)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = utc_timestamp()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(paths.manifest_path, manifest)
        raise
    finally:
        if previous_mplconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mplconfig
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute all production integrals from a newly created empty "
            "cache. Existing run directories are refused."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "New directory to create. If omitted, create a UTC timestamped "
            "directory below fresh_runs/."
        ),
    )
    return parser.parse_args()


def main() -> FreshRunPaths:
    args = parse_args()
    run_dir = default_run_dir() if args.run_dir is None else args.run_dir
    return run_fresh(run_dir)


if __name__ == "__main__":
    main()
