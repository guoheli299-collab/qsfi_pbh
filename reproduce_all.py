#!/usr/bin/env python3
"""Run the complete QSFI-PBH data and figure reproduction workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PACKAGE_DIR = Path(__file__).resolve().parent


def build_commands(
    *,
    package_dir: Path,
    figures_only: bool,
) -> tuple[tuple[str, ...], ...]:
    """Return the deterministic stage commands for the production workflow."""

    commands: list[tuple[str, ...]] = []
    if not figures_only:
        commands.append(
            (
                sys.executable,
                str(package_dir / "compute_results.py"),
            )
        )
    else:
        commands.append(
            (
                sys.executable,
                str(package_dir / "make_figures.py"),
            )
        )
    return tuple(commands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute data and render every final QSFI-PBH figure."
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Redraw existing production NPZ data without recomputing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(
        package_dir=PACKAGE_DIR,
        figures_only=bool(args.figures_only),
    )
    for stage, command in enumerate(commands, start=1):
        print(f"[stage {stage}/{len(commands)}] {' '.join(command)}")
        subprocess.run(command, check=True, cwd=PACKAGE_DIR)


if __name__ == "__main__":
    main()
