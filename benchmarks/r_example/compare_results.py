#!/usr/bin/env python3
"""Compare R and pybspcov benchmark summaries without invoking R."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

STATISTICS = ("posterior_mean", "posterior_sd", "q025", "q50", "q975")
STANDARD_ERROR_MULTIPLIER = 6.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def _finite_float(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _comparison(
    r_row: dict[str, str],
    pybspcov_row: dict[str, str],
    statistic: str,
) -> dict[str, Any]:
    r_value = _finite_float(r_row, statistic)
    pybspcov_value = _finite_float(pybspcov_row, statistic)
    r_mcse = _finite_float(r_row, f"{statistic}_mcse")
    pybspcov_mcse = _finite_float(pybspcov_row, f"{statistic}_mcse")
    result: dict[str, Any] = {
        "r_value": r_value,
        "pybspcov_value": pybspcov_value,
        "r_mcse": r_mcse,
        "pybspcov_mcse": pybspcov_mcse,
    }
    if r_value is None or pybspcov_value is None:
        result.update(
            signed_difference=None,
            absolute_difference=None,
            combined_mcse=None,
            tolerance=None,
            within_tolerance=False,
            reason="missing or nonfinite posterior summary",
        )
        return result
    signed_difference = pybspcov_value - r_value
    result["signed_difference"] = signed_difference
    result["absolute_difference"] = abs(signed_difference)
    if r_mcse is None or pybspcov_mcse is None or r_mcse < 0 or pybspcov_mcse < 0:
        result.update(
            combined_mcse=None,
            tolerance=None,
            within_tolerance=False,
            reason="missing or nonfinite Monte Carlo standard error",
        )
        return result
    combined_mcse = math.hypot(r_mcse, pybspcov_mcse)
    tolerance = STANDARD_ERROR_MULTIPLIER * combined_mcse
    result.update(
        combined_mcse=combined_mcse,
        tolerance=tolerance,
        within_tolerance=abs(signed_difference) <= tolerance,
        reason=None,
    )
    return result


def _keyed_rows(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    keyed: dict[tuple[int, int], dict[str, str]] = {}
    for row in _read_csv(path):
        try:
            key = (int(row["row"]), int(row["column"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} has an invalid row/column key") from error
        if key in keyed:
            raise ValueError(f"{path} contains duplicate covariance key {key}")
        keyed[key] = row
    return keyed


def _timing_categories(path: Path) -> dict[str, float | None]:
    row = _read_csv(path)[0]
    timings: dict[str, float | None] = {}
    for field in row:
        if field == "implementation":
            continue
        value = _finite_float(row, field)
        if value is None or value < 0:
            raise ValueError(f"{path} has invalid timing field {field}")
        timings[field] = value
    return timings


def _validate_summary_rows(
    path: Path, rows: dict[tuple[int, int], dict[str, str]]
) -> None:
    rmse_values = {_finite_float(row, "rmse") for row in rows.values()}
    rmse_mcse_values = {_finite_float(row, "rmse_mcse") for row in rows.values()}
    if None in rmse_values or len(rmse_values) != 1:
        raise ValueError(f"{path} contains inconsistent rmse values")
    if (
        None in rmse_mcse_values
        or len(rmse_mcse_values) != 1
        or next(iter(rmse_mcse_values)) < 0
    ):
        raise ValueError(f"{path} contains inconsistent rmse_mcse values")


def compare_files(
    *,
    r_summary_path: Path,
    pybspcov_summary_path: Path,
    r_timing_path: Path,
    pybspcov_timing_path: Path,
) -> dict[str, Any]:
    """Return a JSON-compatible statistical and timing comparison."""
    r_rows = _keyed_rows(r_summary_path)
    pybspcov_rows = _keyed_rows(pybspcov_summary_path)
    if r_rows.keys() != pybspcov_rows.keys():
        raise ValueError("R and pybspcov summaries contain different covariance keys")
    _validate_summary_rows(r_summary_path, r_rows)
    _validate_summary_rows(pybspcov_summary_path, pybspcov_rows)
    for key in r_rows:
        r_truth = _finite_float(r_rows[key], "truth")
        pybspcov_truth = _finite_float(pybspcov_rows[key], "truth")
        if r_truth is None or pybspcov_truth is None or r_truth != pybspcov_truth:
            raise ValueError(f"truth differs for covariance key {key}")

    posterior_comparisons = []
    all_within_tolerance = True
    for row, column in sorted(r_rows):
        statistics = {
            statistic: _comparison(
                r_rows[(row, column)],
                pybspcov_rows[(row, column)],
                statistic,
            )
            for statistic in STATISTICS
        }
        all_within_tolerance &= all(
            comparison["within_tolerance"] for comparison in statistics.values()
        )
        posterior_comparisons.append(
            {"row": row, "column": column, "statistics": statistics}
        )

    first_key = min(r_rows)
    rmse_comparison = _comparison(r_rows[first_key], pybspcov_rows[first_key], "rmse")
    all_within_tolerance &= rmse_comparison["within_tolerance"]
    return {
        "schema_version": 1,
        "statistical_verdict": "pass" if all_within_tolerance else "fail",
        "standard_error_multiplier": STANDARD_ERROR_MULTIPLIER,
        "posterior_comparisons": posterior_comparisons,
        "rmse_comparison": rmse_comparison,
        "timing_categories": {
            "bspcov": _timing_categories(r_timing_path),
            "pybspcov": _timing_categories(pybspcov_timing_path),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-summary", type=Path, required=True)
    parser.add_argument("--pybspcov-summary", type=Path, required=True)
    parser.add_argument("--r-timing", type=Path, required=True)
    parser.add_argument("--pybspcov-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = compare_files(
        r_summary_path=arguments.r_summary,
        pybspcov_summary_path=arguments.pybspcov_summary,
        r_timing_path=arguments.r_timing,
        pybspcov_timing_path=arguments.pybspcov_timing,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
