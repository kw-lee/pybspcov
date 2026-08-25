"""Pure validation and aggregation helpers for the R/Python benchmark."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PARITY_STATISTICS = ("posterior_mean", "posterior_sd", "q025", "q50", "q975")
REQUIRED_METHODS = {"bm", "sbm", "bandppp", "thresholdppp"}


def _positive_finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the pinned benchmark protocol manifest."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("benchmark manifest must contain a JSON object")
    if parsed.get("schema_version") != "1.0":
        raise ValueError("benchmark manifest must use schema version 1.0")
    if parsed.get("bspcov_version") != "1.0.3":
        raise ValueError("benchmark manifest must pin bspcov 1.0.3")
    if parsed.get("dimensions") != [50, 100, 200]:
        raise ValueError("benchmark dimensions must be [50, 100, 200]")
    methods = parsed.get("methods")
    if not isinstance(methods, dict) or set(methods) != REQUIRED_METHODS:
        raise ValueError("benchmark manifest must define all four methods")
    timing = parsed.get("timing")
    if not isinstance(timing, dict):
        raise TypeError("benchmark timing policy must be an object")
    if timing.get("warm_repetitions") != 3 or timing.get("noisy_warm_repetitions") != 5:
        raise ValueError("benchmark timing policy must use adaptive 3-to-5 repeats")
    return parsed


def combined_mcse_comparison(
    *,
    r_value: object,
    r_mcse: object,
    python_value: object,
    python_mcse: object,
    multiplier: float = 6.0,
) -> dict[str, object]:
    """Compare independent Monte Carlo estimates using combined uncertainty."""
    values = np.asarray([r_value, r_mcse, python_value, python_mcse, multiplier])
    if not np.all(np.isfinite(values)):
        raise ValueError("comparison values and Monte Carlo errors must be finite")
    if float(r_mcse) < 0.0 or float(python_mcse) < 0.0 or multiplier <= 0.0:
        raise ValueError("Monte Carlo errors and multiplier must be non-negative")
    difference = abs(float(python_value) - float(r_value))
    tolerance = float(multiplier * math.hypot(float(r_mcse), float(python_mcse)))
    return {
        "absolute_difference": difference,
        "tolerance": tolerance,
        "within_tolerance": difference <= tolerance,
    }


def _numeric_array(summary: Mapping[str, object], field: str) -> np.ndarray:
    try:
        array = np.asarray(summary[field], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite numeric array") from error
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{field} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite numeric array")
    return array


def compare_parity_summaries(
    r_summary: Mapping[str, object],
    python_summary: Mapping[str, object],
    *,
    multiplier: float = 6.0,
) -> dict[str, object]:
    """Compare complete posterior summaries from independent R and JAX draws."""
    r_truth = _numeric_array(r_summary, "truth")
    python_truth = _numeric_array(python_summary, "truth")
    if not np.array_equal(r_truth, python_truth):
        raise ValueError("R and Python truth matrices differ")

    verdict = True
    statistics_result: dict[str, object] = {}
    for statistic in PARITY_STATISTICS:
        r_values = _numeric_array(r_summary, statistic)
        python_values = _numeric_array(python_summary, statistic)
        r_mcse = _numeric_array(r_summary, f"{statistic}_mcse")
        python_mcse = _numeric_array(python_summary, f"{statistic}_mcse")
        if not all(
            array.shape == r_truth.shape
            for array in (r_values, python_values, r_mcse, python_mcse)
        ):
            raise ValueError(f"{statistic} summary shapes must match truth")
        rows: list[list[dict[str, object]]] = []
        for row in range(r_truth.shape[0]):
            comparisons = []
            for column in range(r_truth.shape[1]):
                comparison = combined_mcse_comparison(
                    r_value=r_values[row, column],
                    r_mcse=r_mcse[row, column],
                    python_value=python_values[row, column],
                    python_mcse=python_mcse[row, column],
                    multiplier=multiplier,
                )
                verdict &= bool(comparison["within_tolerance"])
                comparisons.append(comparison)
            rows.append(comparisons)
        statistics_result[statistic] = rows

    rmse_comparison = combined_mcse_comparison(
        r_value=r_summary.get("rmse"),
        r_mcse=r_summary.get("rmse_mcse"),
        python_value=python_summary.get("rmse"),
        python_mcse=python_summary.get("rmse_mcse"),
        multiplier=multiplier,
    )
    verdict &= bool(rmse_comparison["within_tolerance"])
    return {
        "verdict": "pass" if verdict else "fail",
        "multiplier": multiplier,
        "statistics": statistics_result,
        "rmse": rmse_comparison,
    }


def needs_additional_repetitions(
    times: Sequence[float], *, threshold: float = 0.1
) -> bool:
    """Return whether a three-run cell needs two additional repetitions."""
    values = [_positive_finite(value, "warm timing") for value in times]
    if len(values) == 5:
        return False
    if len(values) != 3:
        raise ValueError("adaptive timing requires three or five repetitions")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("relative range threshold must be finite and non-negative")
    median = statistics.median(values)
    return (max(values) - min(values)) / median > threshold


def summarize_timings(
    times: Sequence[float], *, retained_draws: int
) -> dict[str, object]:
    """Summarize a complete three- or five-repetition warm timing cell."""
    values = [_positive_finite(value, "warm timing") for value in times]
    if len(values) not in {3, 5}:
        raise ValueError("warm timing must contain three or five repetitions")
    if isinstance(retained_draws, bool) or retained_draws <= 0:
        raise ValueError("retained_draws must be a positive integer")
    median = float(statistics.median(values))
    return {
        "repetitions": len(values),
        "median_seconds": median,
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "retained_draws_per_second": retained_draws / median,
    }


def select_headline_dtype(
    *,
    float64_seconds: float,
    float32_seconds: float,
    float64_parity: bool,
    float32_parity: bool,
) -> str:
    """Choose float32 only when validated and at least five percent faster."""
    time64 = _positive_finite(float64_seconds, "float64 timing")
    time32 = _positive_finite(float32_seconds, "float32 timing")
    if not float64_parity:
        raise ValueError("float64 parity must pass before publishing performance")
    if float32_parity and time32 <= 0.95 * time64:
        return "float32"
    return "float64"


def geometric_mean(values: Sequence[float]) -> float:
    """Return a stable geometric mean for finite positive ratios."""
    checked = [_positive_finite(value, "ratio") for value in values]
    if not checked:
        raise ValueError("at least one ratio is required")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def validate_timing_record(record: Mapping[str, object]) -> None:
    """Reject benchmark records that are unsafe to publish."""
    if record.get("schema_version") != "1.0":
        raise ValueError("timing records must use schema version 1.0")
    if record.get("method") not in REQUIRED_METHODS:
        raise ValueError("timing record has an unknown method")
    if record.get("git_dirty") is not False:
        raise ValueError("publishable timing records require a clean git worktree")
    revision = str(record.get("git_revision", ""))
    if len(revision) != 40:
        raise ValueError("timing record requires a full git revision")
    fixture_hash = str(record.get("fixture_sha256", ""))
    if len(fixture_hash) != 64:
        raise ValueError("timing record requires a SHA-256 fixture hash")
    if record.get("device") == "gpu" and record.get("actual_platform") != "cuda":
        raise ValueError("GPU records must have an actual CUDA platform")
    for field in (
        "posterior_mean_finite",
        "posterior_mean_symmetric",
        "posterior_mean_spd",
    ):
        if record.get(field) is not True:
            raise ValueError(f"timing record failed {field}")
    if record.get("rejected_sweeps") != 0:
        raise ValueError("timing record contains rejected sweeps")
    _positive_finite(record.get("cold_end_to_end_seconds"), "cold timing")
    retained_draws = record.get("retained_draws")
    if isinstance(retained_draws, bool) or not isinstance(retained_draws, int):
        raise TypeError("retained_draws must be a positive integer")
    summarize_timings(record.get("warm_seconds", []), retained_draws=retained_draws)
