import csv
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_DIR = Path(__file__).parents[1] / "benchmarks" / "r_example"
sys.path.insert(0, str(BENCHMARK_DIR))

from compare_results import compare_files
from run_pybspcov import (
    detect_cpu_model,
    summarize_draws,
    validate_chain_output,
)

SUMMARY_FIELDS = (
    "implementation",
    "row",
    "column",
    "posterior_mean",
    "posterior_mean_mcse",
    "posterior_sd",
    "posterior_sd_mcse",
    "q025",
    "q025_mcse",
    "q50",
    "q50_mcse",
    "q975",
    "q975_mcse",
    "truth",
    "rmse",
    "rmse_mcse",
)


def _summary_rows(implementation: str) -> list[dict[str, object]]:
    return [
        {
            "implementation": implementation,
            "row": 1,
            "column": 1,
            "posterior_mean": 1.00,
            "posterior_mean_mcse": 0.01,
            "posterior_sd": 0.40,
            "posterior_sd_mcse": 0.01,
            "q025": 0.30,
            "q025_mcse": 0.01,
            "q50": 0.95,
            "q50_mcse": 0.01,
            "q975": 1.80,
            "q975_mcse": 0.01,
            "truth": 1.0,
            "rmse": 0.10,
            "rmse_mcse": 0.02,
        },
        {
            "implementation": implementation,
            "row": 1,
            "column": 2,
            "posterior_mean": -0.20,
            "posterior_mean_mcse": 0.01,
            "posterior_sd": 0.25,
            "posterior_sd_mcse": 0.01,
            "q025": -0.70,
            "q025_mcse": 0.01,
            "q50": -0.18,
            "q50_mcse": 0.01,
            "q975": 0.25,
            "q975_mcse": 0.01,
            "truth": 0.0,
            "rmse": 0.10,
            "rmse_mcse": 0.02,
        },
    ]


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(
    directory: Path,
    pybspcov_rows: list[dict[str, object]],
    *,
    r_sampler_seconds: float = 10.0,
    pybspcov_steady_state_seconds: float = 2.0,
) -> tuple[Path, Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    r_summary = directory / "r_summary.csv"
    pybspcov_summary = directory / "pybspcov_summary.csv"
    r_timing = directory / "r_timing.csv"
    pybspcov_timing = directory / "pybspcov_timing.csv"
    _write_csv(r_summary, SUMMARY_FIELDS, _summary_rows("bspcov"))
    _write_csv(pybspcov_summary, SUMMARY_FIELDS, pybspcov_rows)
    _write_csv(
        r_timing,
        ("implementation", "sampler_seconds", "end_to_end_seconds"),
        [
            {
                "implementation": "bspcov",
                "sampler_seconds": r_sampler_seconds,
                "end_to_end_seconds": r_sampler_seconds + 1.0,
            }
        ],
    )
    _write_csv(
        pybspcov_timing,
        (
            "implementation",
            "compile_plus_execution_seconds",
            "steady_state_seconds",
            "end_to_end_seconds",
        ),
        [
            {
                "implementation": "pybspcov",
                "compile_plus_execution_seconds": 5.0,
                "steady_state_seconds": pybspcov_steady_state_seconds,
                "end_to_end_seconds": 8.0,
            }
        ],
    )
    return r_summary, pybspcov_summary, r_timing, pybspcov_timing


def _compare(paths: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    r_summary, pybspcov_summary, r_timing, pybspcov_timing = paths
    return compare_files(
        r_summary_path=r_summary,
        pybspcov_summary_path=pybspcov_summary,
        r_timing_path=r_timing,
        pybspcov_timing_path=pybspcov_timing,
    )


def test_in_tolerance_posterior_summaries_pass(tmp_path: Path) -> None:
    pybspcov_rows = _summary_rows("pybspcov")
    for row in pybspcov_rows:
        for statistic in ("posterior_mean", "posterior_sd", "q025", "q50", "q975"):
            row[statistic] = float(row[statistic]) + 0.05
        row["rmse"] = 0.14

    result = _compare(_write_fixture(tmp_path, pybspcov_rows))

    assert result["statistical_verdict"] == "pass"
    first_mean = result["posterior_comparisons"][0]["statistics"]["posterior_mean"]
    assert first_mean["signed_difference"] == pytest.approx(0.05)
    assert first_mean["absolute_difference"] == pytest.approx(0.05)
    assert first_mean["tolerance"] == pytest.approx(6.0 * (0.01**2 + 0.01**2) ** 0.5)
    assert first_mean["within_tolerance"] is True
    assert result["rmse_comparison"]["within_tolerance"] is True


def test_out_of_tolerance_posterior_summary_fails(tmp_path: Path) -> None:
    pybspcov_rows = _summary_rows("pybspcov")
    pybspcov_rows[1]["q975"] = 0.45

    result = _compare(_write_fixture(tmp_path, pybspcov_rows))

    assert result["statistical_verdict"] == "fail"
    failed_quantile = result["posterior_comparisons"][1]["statistics"]["q975"]
    assert failed_quantile["absolute_difference"] == pytest.approx(0.20)
    assert failed_quantile["within_tolerance"] is False


def test_timing_fields_do_not_determine_statistical_verdict(tmp_path: Path) -> None:
    pybspcov_rows = _summary_rows("pybspcov")
    fast_paths = _write_fixture(
        tmp_path / "fast",
        pybspcov_rows,
        r_sampler_seconds=100.0,
        pybspcov_steady_state_seconds=0.001,
    )
    slow_paths = _write_fixture(
        tmp_path / "slow",
        pybspcov_rows,
        r_sampler_seconds=0.001,
        pybspcov_steady_state_seconds=100.0,
    )

    fast_result = _compare(fast_paths)
    slow_result = _compare(slow_paths)

    assert fast_result["statistical_verdict"] == "pass"
    assert slow_result["statistical_verdict"] == "pass"
    assert fast_result["posterior_comparisons"] == slow_result["posterior_comparisons"]
    assert fast_result["timing_categories"] != slow_result["timing_categories"]


@pytest.mark.parametrize("invalid_mcse", ["", "nan", "inf"])
def test_missing_or_nonfinite_mcse_fails_verdict(
    tmp_path: Path, invalid_mcse: str
) -> None:
    pybspcov_rows = _summary_rows("pybspcov")
    pybspcov_rows[0]["posterior_mean_mcse"] = invalid_mcse

    result = _compare(_write_fixture(tmp_path, pybspcov_rows))

    assert result["statistical_verdict"] == "fail"
    mean = result["posterior_comparisons"][0]["statistics"]["posterior_mean"]
    assert mean["within_tolerance"] is False
    assert mean["reason"] == "missing or nonfinite Monte Carlo standard error"


def test_mismatched_truth_is_rejected(tmp_path: Path) -> None:
    pybspcov_rows = _summary_rows("pybspcov")
    pybspcov_rows[1]["truth"] = 0.1

    with pytest.raises(ValueError, match="truth differs"):
        _compare(_write_fixture(tmp_path, pybspcov_rows))


@pytest.mark.parametrize(
    ("field", "invalid_value"), [("rmse", 0.11), ("rmse_mcse", 0.03)]
)
def test_inconsistent_repeated_rmse_is_rejected(
    tmp_path: Path, field: str, invalid_value: float
) -> None:
    pybspcov_rows = _summary_rows("pybspcov")
    pybspcov_rows[1][field] = invalid_value

    with pytest.raises(ValueError, match=f"inconsistent {field}"):
        _compare(_write_fixture(tmp_path, pybspcov_rows))


@pytest.mark.parametrize("invalid_timing", [-1.0, float("nan"), float("inf")])
def test_invalid_timing_is_rejected_without_becoming_a_verdict(
    tmp_path: Path, invalid_timing: float
) -> None:
    paths = _write_fixture(
        tmp_path,
        _summary_rows("pybspcov"),
        r_sampler_seconds=invalid_timing,
    )

    with pytest.raises(ValueError, match="timing"):
        _compare(paths)


def test_batch_mcse_summary_uses_deterministic_contiguous_batches() -> None:
    draws = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 2.0], [2.0, 3.0]],
            [[5.0, 4.0], [4.0, 5.0]],
            [[7.0, 6.0], [6.0, 7.0]],
        ]
    )
    truth = np.array([[4.0, 0.0], [0.0, 4.0]])

    summary = summarize_draws(draws, truth)

    np.testing.assert_allclose(summary["posterior_mean"], [[4.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(summary["posterior_mean_mcse"], np.full((2, 2), 2.0))
    np.testing.assert_allclose(summary["posterior_sd_mcse"], np.zeros((2, 2)))
    np.testing.assert_allclose(summary["q025_mcse"], np.full((2, 2), 2.0))
    np.testing.assert_allclose(summary["q50_mcse"], np.full((2, 2), 2.0))
    np.testing.assert_allclose(summary["q975_mcse"], np.full((2, 2), 2.0))
    assert summary["rmse"] == pytest.approx(4.5**0.5)
    assert summary["rmse_mcse"] == pytest.approx((14.5**0.5 - 2.5**0.5) / 2.0)


def test_batch_mcse_summary_requires_four_draws() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        summarize_draws(np.ones((3, 2, 2)), np.eye(2))


def test_chain_validation_rejects_a_failed_sweep() -> None:
    with pytest.raises(RuntimeError, match="rejected sweep"):
        validate_chain_output(np.array([True, False]), np.ones((2, 2, 2)))


def test_chain_validation_rejects_nonfinite_draws() -> None:
    draws = np.ones((2, 2, 2))
    draws[1, 0, 0] = np.nan

    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_chain_output(np.array([True, True]), draws)


def test_cpu_model_detection_returns_a_nonempty_provenance_value() -> None:
    assert detect_cpu_model().strip()
