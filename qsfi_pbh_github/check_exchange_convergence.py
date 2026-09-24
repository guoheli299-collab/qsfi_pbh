#!/usr/bin/env python3
"""Check the r_s=3, nu_tilde=0.1 exchange coefficient under refinement.

This is an opt-in numerical diagnostic. It computes only the scalar-exchange
projection and writes into an isolated directory. The production workflow
does not call it; unit tests exercise its comparison and output logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping

from qsfi_pbh.backend import make_args
from qsfi_pbh.cumulants import (
    FullMomentumIntegrationSettings,
    StageEstimate,
    estimate_stage,
)
from qsfi_pbh.momentum_shapes import (
    FullMomentumShapeEvaluator,
    build_ratio_function_bundle,
)
from qsfi_pbh.propagator import build_grid, full_momentum_settings_from_args
from qsfi_pbh.sound_speed import SOUND_SPEED_CASES, SoundSpeedConfiguration


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "convergence_results" / "r3_nu0p1_exchange"
)
REPORT_NAME = "r3_nu0p1_exchange_convergence.json"
NU_TILDE = 0.1
DEFAULT_TOLERANCE = 0.03
FIXED_REPLICATES = 6


@dataclass(frozen=True)
class NumericalCase:
    """One controlled exchange-projection resolution."""

    label: str
    sobol_power: int
    ratio_nodes: int


CASES = (
    NumericalCase("m12_n128", sobol_power=12, ratio_nodes=128),
    NumericalCase("m12_n256", sobol_power=12, ratio_nodes=256),
    NumericalCase("m13_n256", sobol_power=13, ratio_nodes=256),
    NumericalCase("m14_n256", sobol_power=14, ratio_nodes=256),
)


def _r3_sound_speed() -> SoundSpeedConfiguration:
    matches = [case for case in SOUND_SPEED_CASES if math.isclose(case.ratio, 3.0)]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one production sound-speed case with r_s=3.")
    return matches[0]


def _positive_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be positive and finite.")
    return value


def _relative_difference(first: float, second: float) -> float:
    first = float(first)
    second = float(second)
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("Compared coefficients must be finite.")
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return 0.0
    return abs(second - first) / scale


def build_comparisons(
    estimates: Mapping[str, Mapping[str, object]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, dict[str, object]]:
    """Compare grid and Sobol refinements using the hatted coefficient."""

    limit = _positive_tolerance(tolerance)
    pairs = (
        (
            "ratio_grid_128_to_256_at_m12",
            "m12_n128",
            "m12_n256",
        ),
        (
            "sobol_m12_to_m13_at_n256",
            "m12_n256",
            "m13_n256",
        ),
        (
            "sobol_m13_to_m14_at_n256",
            "m13_n256",
            "m14_n256",
        ),
        (
            "overall_m12_n128_to_m14_n256",
            "m12_n128",
            "m14_n256",
        ),
    )
    comparisons: dict[str, dict[str, object]] = {}
    for name, first_label, second_label in pairs:
        try:
            first = float(estimates[first_label]["C4_exchange_hat"])
            second = float(estimates[second_label]["C4_exchange_hat"])
        except KeyError as error:
            raise ValueError(f"Missing convergence estimate field: {error}") from error
        difference = _relative_difference(first, second)
        comparisons[name] = {
            "from": first_label,
            "to": second_label,
            "relative_difference": difference,
            "tolerance": limit,
            "passed": bool(difference <= limit),
        }
    return comparisons


def _case_settings(
    base: FullMomentumIntegrationSettings,
    case: NumericalCase,
) -> FullMomentumIntegrationSettings:
    return replace(
        base,
        ratio_nodes=case.ratio_nodes,
        sobol_power_exchange=case.sobol_power,
        sobol_min_replicates_four_point=FIXED_REPLICATES,
        sobol_max_replicates_four_point=FIXED_REPLICATES,
        sobol_max_power_increment_four_point=0,
    )


def _estimate_record(
    case: NumericalCase,
    estimate: StageEstimate,
    *,
    hatted_factor: float,
) -> dict[str, object]:
    return {
        "label": case.label,
        "sobol_power": case.sobol_power,
        "ratio_nodes": case.ratio_nodes,
        "S4_per_lambda3_over_H_squared_exchange": estimate.coefficient,
        "S4_per_lambda3_over_H_squared_exchange_standard_error": (
            estimate.coefficient_standard_error
        ),
        "C4_exchange_hat": estimate.coefficient * hatted_factor,
        "C4_exchange_hat_standard_error": (
            estimate.coefficient_standard_error * abs(hatted_factor)
        ),
        "sampling": estimate.as_dict(),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary_name = handle.name
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def run_convergence(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, object]:
    """Run the four fixed exchange estimates and write their JSON report."""

    limit = _positive_tolerance(tolerance)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    sound_speed = _r3_sound_speed()
    args = make_args(force=False, sound_speed=sound_speed)
    args.cache_dir = str(root / "propagator_cache")
    args.force = False
    grid = build_grid(NU_TILDE, args)
    base_settings = full_momentum_settings_from_args(args)

    evaluator_by_nodes: dict[int, FullMomentumShapeEvaluator] = {}
    estimates: dict[str, dict[str, object]] = {}
    hatted_factor = base_settings.power_area / base_settings.lambda2_over_H**4

    for case in CASES:
        settings = _case_settings(base_settings, case)
        evaluator = evaluator_by_nodes.get(case.ratio_nodes)
        if evaluator is None:
            functions = build_ratio_function_bundle(
                grid,
                minimum_ratio=settings.ratio_floor,
                node_count=case.ratio_nodes,
            )
            evaluator = FullMomentumShapeEvaluator(
                functions,
                nutilde=NU_TILDE,
                sound_speed=settings.sound_speed,
            )
            evaluator_by_nodes[case.ratio_nodes] = evaluator

        checkpoint_path = (
            root
            / "qmc_checkpoints"
            / f"ratio_nodes_{case.ratio_nodes}"
            / "exchange.json"
        )
        estimate = estimate_stage(
            evaluator,
            "exchange",
            settings,
            checkpoint_path=checkpoint_path,
        )
        estimates[case.label] = _estimate_record(
            case,
            estimate,
            hatted_factor=hatted_factor,
        )

    comparisons = build_comparisons(estimates, tolerance=limit)
    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "r_s=3, nu_tilde=0.1 scalar-exchange convergence check",
        "sound_speed": sound_speed.as_dict(),
        "nu_tilde": NU_TILDE,
        "fixed_replicates": FIXED_REPLICATES,
        "adaptive_sobol_power_increment": 0,
        "relative_tolerance": limit,
        "normalization": {
            "power_area": base_settings.power_area,
            "lambda2_over_H": base_settings.lambda2_over_H,
            "barred_to_hatted_exchange_factor": hatted_factor,
        },
        "estimates": estimates,
        "comparisons": comparisons,
        "passed": bool(all(item["passed"] for item in comparisons.values())),
    }
    _write_json(root / REPORT_NAME, report)
    return report


def _print_report(report: Mapping[str, object], report_path: Path) -> None:
    print("\nExchange coefficient convergence: r_s=3, nu_tilde=0.1")
    print("case       C4_exchange_hat       standard error      relative SE")
    estimates = report["estimates"]
    assert isinstance(estimates, Mapping)
    for case in CASES:
        record = estimates[case.label]
        assert isinstance(record, Mapping)
        sampling = record["sampling"]
        assert isinstance(sampling, Mapping)
        print(
            f"{case.label:<10} "
            f"{float(record['C4_exchange_hat']): .10e}  "
            f"{float(record['C4_exchange_hat_standard_error']):.3e}  "
            f"{float(sampling['relative_standard_error']):.3%}"
        )

    print("\nRefinement shifts")
    comparisons = report["comparisons"]
    assert isinstance(comparisons, Mapping)
    for name, raw_comparison in comparisons.items():
        assert isinstance(raw_comparison, Mapping)
        status = "PASS" if raw_comparison["passed"] else "FAIL"
        print(
            f"{status:4}  {name}: "
            f"{float(raw_comparison['relative_difference']):.3%} "
            f"(limit {float(raw_comparison['tolerance']):.3%})"
        )
    print(f"\nReport: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the r_s=3, nu_tilde=0.1 exchange coefficient at Sobol "
            "powers 12--14 and 128/256 propagator-ratio nodes."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="isolated cache and report directory",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="maximum accepted relative refinement shift (default: 0.03)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_convergence(args.output_dir, tolerance=args.tolerance)
    report_path = args.output_dir.expanduser().resolve() / REPORT_NAME
    _print_report(report, report_path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
