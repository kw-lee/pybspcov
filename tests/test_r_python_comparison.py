import importlib.util
import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "r_comparison"
CORE_PATH = BENCHMARK_DIR / "core.py"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"


def _core():
    assert CORE_PATH.is_file(), "benchmarks/r_comparison/core.py is required"
    spec = importlib.util.spec_from_file_location("r_comparison_core", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_pins_complete_four_method_matrix() -> None:
    core = _core()

    manifest = core.load_manifest(MANIFEST_PATH)

    assert manifest["schema_version"] == "1.0"
    assert manifest["bspcov_version"] == "1.0.3"
    assert manifest["seed"] == 20260803
    assert manifest["dimensions"] == [50, 100, 200]
    assert manifest["n_factor"] == 3
    assert set(manifest["methods"]) == {"bm", "sbm", "bandppp", "thresholdppp"}
    assert manifest["methods"]["bm"] == {"burnin": 50, "samples": 50}
    assert manifest["methods"]["sbm"] == {
        "burnin": 50,
        "samples": 50,
        "cutoff_method": "correlation",
        "retained_fraction": 0.2,
    }
    assert manifest["methods"]["bandppp"] == {
        "samples_per_batch": 100,
        "epsilon": 0.05,
        "bandwidth_divisor": 20,
    }
    assert manifest["methods"]["thresholdppp"] == {
        "samples_per_batch": 100,
        "threshold": 0.1,
        "method": "hard",
        "epsilon": 0.1,
    }
    assert manifest["optimized"]["parallelism"] == 8
    assert manifest["optimized"]["cpu_cores"] == 8
    assert manifest["cpu_baseline"]["parallelism"] == 1
    assert manifest["cpu_baseline"]["cpu_cores"] == 1
    assert manifest["timing"] == {
        "cold_repetitions": 1,
        "warm_repetitions": 3,
        "noisy_warm_repetitions": 5,
        "relative_range_threshold": 0.1,
    }


def test_combined_mcse_comparison_uses_six_standard_errors() -> None:
    core = _core()

    comparison = core.combined_mcse_comparison(
        r_value=1.0,
        r_mcse=0.03,
        python_value=1.2,
        python_mcse=0.04,
    )

    assert comparison == {
        "absolute_difference": pytest.approx(0.2),
        "tolerance": pytest.approx(0.3),
        "within_tolerance": True,
    }


def test_combined_mcse_comparison_rejects_missing_or_nonfinite_inputs() -> None:
    core = _core()

    with pytest.raises(ValueError, match="finite"):
        core.combined_mcse_comparison(
            r_value=1.0,
            r_mcse=math.nan,
            python_value=1.0,
            python_mcse=0.1,
        )


def test_parity_summary_requires_every_statistic_to_pass() -> None:
    core = _core()
    r_summary = {
        "truth": [[1.0, 0.0], [0.0, 1.0]],
        "posterior_mean": [[1.0, 0.0], [0.0, 1.0]],
        "posterior_mean_mcse": [[0.01, 0.01], [0.01, 0.01]],
        "posterior_sd": [[0.2, 0.1], [0.1, 0.2]],
        "posterior_sd_mcse": [[0.01, 0.01], [0.01, 0.01]],
        "q025": [[0.7, -0.2], [-0.2, 0.7]],
        "q025_mcse": [[0.01, 0.01], [0.01, 0.01]],
        "q50": [[1.0, 0.0], [0.0, 1.0]],
        "q50_mcse": [[0.01, 0.01], [0.01, 0.01]],
        "q975": [[1.3, 0.2], [0.2, 1.3]],
        "q975_mcse": [[0.01, 0.01], [0.01, 0.01]],
        "rmse": 0.0,
        "rmse_mcse": 0.01,
    }
    python_summary = json.loads(json.dumps(r_summary))
    python_summary["posterior_mean"][0][1] = 0.086

    result = core.compare_parity_summaries(r_summary, python_summary)

    assert result["verdict"] == "fail"
    assert result["statistics"]["posterior_mean"][0][1]["within_tolerance"] is False
    assert result["statistics"]["q50"][0][1]["within_tolerance"] is True


def test_parity_summary_rejects_different_truth() -> None:
    core = _core()
    summary = {
        "truth": [[1.0]],
        "posterior_mean": [[1.0]],
        "posterior_mean_mcse": [[0.1]],
        "posterior_sd": [[0.1]],
        "posterior_sd_mcse": [[0.1]],
        "q025": [[0.8]],
        "q025_mcse": [[0.1]],
        "q50": [[1.0]],
        "q50_mcse": [[0.1]],
        "q975": [[1.2]],
        "q975_mcse": [[0.1]],
        "rmse": 0.0,
        "rmse_mcse": 0.1,
    }
    other = json.loads(json.dumps(summary))
    other["truth"] = [[2.0]]

    with pytest.raises(ValueError, match="truth"):
        core.compare_parity_summaries(summary, other)


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ([9.8, 10.0, 10.2], False),
        ([8.0, 10.0, 11.0], True),
        ([8.0, 10.0, 11.0, 9.5, 10.5], False),
    ],
)
def test_adaptive_repetition_rule(times: list[float], expected: bool) -> None:
    core = _core()

    assert core.needs_additional_repetitions(times, threshold=0.1) is expected


def test_timing_summary_reports_literal_median_range_and_throughput() -> None:
    core = _core()

    result = core.summarize_timings([4.0, 2.0, 3.0], retained_draws=600)

    assert result == {
        "repetitions": 3,
        "median_seconds": 3.0,
        "minimum_seconds": 2.0,
        "maximum_seconds": 4.0,
        "retained_draws_per_second": 200.0,
    }


@pytest.mark.parametrize("times", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_timing_summary_rejects_non_protocol_repeat_counts(times: list[float]) -> None:
    core = _core()

    with pytest.raises(ValueError, match="three or five"):
        core.summarize_timings(times, retained_draws=100)


def test_headline_dtype_requires_parity_and_five_percent_speedup() -> None:
    core = _core()

    assert core.select_headline_dtype(
        float64_seconds=10.0,
        float32_seconds=9.5,
        float64_parity=True,
        float32_parity=True,
    ) == "float32"
    assert core.select_headline_dtype(
        float64_seconds=10.0,
        float32_seconds=9.51,
        float64_parity=True,
        float32_parity=True,
    ) == "float64"
    assert core.select_headline_dtype(
        float64_seconds=10.0,
        float32_seconds=8.0,
        float64_parity=True,
        float32_parity=False,
    ) == "float64"


def test_headline_dtype_rejects_failed_float64_parity() -> None:
    core = _core()

    with pytest.raises(ValueError, match="float64 parity"):
        core.select_headline_dtype(
            float64_seconds=10.0,
            float32_seconds=8.0,
            float64_parity=False,
            float32_parity=True,
        )


def test_geometric_mean_uses_all_scaling_cells() -> None:
    core = _core()

    assert core.geometric_mean([2.0, 4.0, 8.0]) == pytest.approx(4.0)


def test_timing_record_rejects_dirty_or_fallback_runs() -> None:
    core = _core()
    record = {
        "schema_version": "1.0",
        "method": "bm",
        "dimension": 200,
        "n_observations": 600,
        "seed": 20260803,
        "fixture_sha256": "a" * 64,
        "implementation": "pybspcov",
        "version": "0.1.0.dev0",
        "device": "gpu",
        "actual_platform": "cuda",
        "dtype": "float32",
        "execution": "vmap",
        "parallelism": 8,
        "cpu_cores": 8,
        "retained_draws": 400,
        "cold_end_to_end_seconds": 12.0,
        "warm_seconds": [3.0, 3.1, 3.2],
        "posterior_mean_finite": True,
        "posterior_mean_symmetric": True,
        "posterior_mean_spd": True,
        "rejected_sweeps": 0,
        "git_revision": "b" * 40,
        "git_dirty": False,
    }

    core.validate_timing_record(record)

    dirty = dict(record, git_dirty=True)
    with pytest.raises(ValueError, match="clean git"):
        core.validate_timing_record(dirty)
    fallback = dict(record, actual_platform="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        core.validate_timing_record(fallback)
