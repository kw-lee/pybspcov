import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "r_comparison"
CORE_PATH = BENCHMARK_DIR / "core.py"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
FIXTURES_PATH = BENCHMARK_DIR / "fixtures.py"
PYTHON_RUNNER_PATH = BENCHMARK_DIR / "run_pybspcov.py"
MATRIX_PATH = BENCHMARK_DIR / "run_matrix.py"
R_RUNNER_PATH = BENCHMARK_DIR / "run_bspcov.R"
R_HASH_HELPER_PATH = BENCHMARK_DIR / "fixture_hash.R"
PARITY_RUNNER_PATH = BENCHMARK_DIR / "run_pybspcov_parity.py"
R_PARITY_RUNNER_PATH = BENCHMARK_DIR / "run_bspcov_parity.R"
PARITY_COMPARE_PATH = BENCHMARK_DIR / "compare_parity.py"

sys.path.insert(0, str(BENCHMARK_DIR))


def _core():
    assert CORE_PATH.is_file(), "benchmarks/r_comparison/core.py is required"
    spec = importlib.util.spec_from_file_location("r_comparison_core", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(path: Path, name: str):
    assert path.is_file(), f"{path.relative_to(PROJECT_ROOT)} is required"
    spec = importlib.util.spec_from_file_location(name, path)
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

    assert (
        core.select_headline_dtype(
            float64_seconds=10.0,
            float32_seconds=9.5,
            float64_parity=True,
            float32_parity=True,
        )
        == "float32"
    )
    assert (
        core.select_headline_dtype(
            float64_seconds=10.0,
            float32_seconds=9.51,
            float64_parity=True,
            float32_parity=True,
        )
        == "float64"
    )
    assert (
        core.select_headline_dtype(
            float64_seconds=10.0,
            float32_seconds=8.0,
            float64_parity=True,
            float32_parity=False,
        )
        == "float64"
    )


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


def test_sparse_fixture_is_deterministic_centered_and_positive_definite() -> None:
    fixtures = _module(FIXTURES_PATH, "r_comparison_fixtures")

    first = fixtures.generate_fixture(
        dimension=5,
        n_observations=15,
        seed=17,
        kind="sparse",
        density=0.2,
    )
    second = fixtures.generate_fixture(
        dimension=5,
        n_observations=15,
        seed=17,
        kind="sparse",
        density=0.2,
    )

    for key in ("observations", "truth_covariance", "initial_covariance"):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(first["observations"].mean(axis=0), 0.0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(first["truth_covariance"]) > 0.0)
    assert np.all(np.linalg.eigvalsh(first["initial_covariance"]) > 0.0)


def test_banded_fixture_has_no_entries_outside_requested_band() -> None:
    fixtures = _module(FIXTURES_PATH, "r_comparison_fixtures_banded")

    fixture = fixtures.generate_fixture(
        dimension=6,
        n_observations=18,
        seed=23,
        kind="banded",
        bandwidth=2,
    )

    truth = fixture["truth_covariance"]
    for row in range(6):
        for column in range(6):
            if abs(row - column) > 2:
                assert truth[row, column] == 0.0
    assert np.all(np.linalg.eigvalsh(truth) > 0.0)


def test_written_fixture_round_trips_with_a_content_hash(tmp_path: Path) -> None:
    fixtures = _module(FIXTURES_PATH, "r_comparison_fixtures_roundtrip")
    fixture = fixtures.generate_fixture(
        dimension=4,
        n_observations=12,
        seed=29,
        kind="sparse",
        density=0.25,
    )

    metadata = fixtures.write_fixture(tmp_path, fixture)
    loaded = fixtures.load_fixture(tmp_path)

    assert metadata == {
        "dimension": 4,
        "n_observations": 12,
        "sha256": fixtures.fixture_sha256(tmp_path),
    }
    assert len(metadata["sha256"]) == 64
    for key in fixture:
        np.testing.assert_allclose(loaded[key], fixture[key], rtol=0.0, atol=1e-15)


def test_r_fixture_hash_matches_python_without_colon_formatting(tmp_path: Path) -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is unavailable")
    fixtures = _module(FIXTURES_PATH, "r_comparison_cross_language_hash")
    fixture = fixtures.generate_fixture(
        dimension=4,
        n_observations=12,
        seed=31,
        kind="sparse",
        density=0.25,
    )
    metadata = fixtures.write_fixture(tmp_path, fixture)

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            (
                "arguments <- commandArgs(trailingOnly=TRUE); "
                "source(arguments[[1L]]); "
                "cat(fixture_sha256(arguments[[2L]]))"
            ),
            str(R_HASH_HELPER_PATH),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == metadata["sha256"]


def test_generated_matrix_fixtures_include_committed_p5_parity_case(
    tmp_path: Path,
) -> None:
    matrix = _module(MATRIX_PATH, "r_comparison_matrix_parity_fixture")
    fixtures = _module(FIXTURES_PATH, "r_comparison_parity_fixture")
    manifest = _core().load_manifest(MANIFEST_PATH)

    matrix.generate_all_fixtures(manifest, tmp_path)
    parity = fixtures.load_fixture(tmp_path / "parity-p5")

    assert parity["observations"].shape == (20, 5)
    np.testing.assert_array_equal(
        parity["truth_covariance"],
        np.loadtxt(
            PROJECT_ROOT / "benchmarks/r_example/data/bm_example_truth.csv",
            delimiter=",",
        ),
    )


@pytest.mark.parametrize(
    ("method", "expected_class", "expected_attributes"),
    [
        ("bm", "BMSPCov", {"burnin": 50, "n_samples": 50}),
        (
            "sbm",
            "SBMSPCov",
            {
                "burnin": 50,
                "n_samples": 50,
                "cutoff_method": "correlation",
                "retained_fraction": 0.2,
            },
        ),
        (
            "bandppp",
            "BandPPP",
            {"bandwidth": 5, "epsilon": 0.05, "n_samples": 100},
        ),
        (
            "thresholdppp",
            "ThresholdPPP",
            {
                "threshold": 0.1,
                "method": "hard",
                "epsilon": 0.1,
                "n_samples": 100,
            },
        ),
    ],
)
def test_python_runner_builds_manifest_pinned_estimators(
    method: str, expected_class: str, expected_attributes: dict[str, object]
) -> None:
    runner = _module(PYTHON_RUNNER_PATH, "r_comparison_python_runner")
    manifest = _core().load_manifest(MANIFEST_PATH)

    estimator = runner.build_estimator(
        method,
        dimension=100,
        dtype="float64",
        device="cpu",
        parallelism=1,
        manifest=manifest,
    )

    assert type(estimator).__name__ == expected_class
    assert estimator.dtype == "float64"
    assert estimator.device == "cpu"
    assert estimator.n_chains == 1
    for attribute, expected in expected_attributes.items():
        assert getattr(estimator, attribute) == expected


def test_python_runner_executes_real_thresholdppp_smoke_cell() -> None:
    runner = _module(PYTHON_RUNNER_PATH, "r_comparison_python_runner_smoke")
    manifest = _core().load_manifest(MANIFEST_PATH)
    fixture = _module(FIXTURES_PATH, "r_comparison_fixtures_smoke").generate_fixture(
        dimension=3,
        n_observations=9,
        seed=31,
        kind="sparse",
        density=0.3,
    )

    result = runner.measure_cell(
        observations=fixture["observations"],
        truth_covariance=fixture["truth_covariance"],
        method="thresholdppp",
        dtype="float32",
        device="cpu",
        parallelism=1,
        manifest=manifest,
        seed=37,
        warm_repetitions=3,
        smoke_samples=2,
    )

    assert result["actual_platform"] == "cpu"
    assert result["retained_draws"] == 2
    assert result["cold_fit_seconds"] > 0.0
    assert len(result["warm_seconds"]) in {3, 5}
    assert result["posterior_mean_finite"] is True
    assert result["posterior_mean_symmetric"] is True
    assert result["posterior_mean_spd"] is True
    assert result["rejected_sweeps"] == 0


def test_matrix_contains_only_the_pre_registered_sixty_cells() -> None:
    matrix = _module(MATRIX_PATH, "r_comparison_matrix")
    manifest = _core().load_manifest(MANIFEST_PATH)

    cells = matrix.build_cells(manifest)

    assert len(cells) == 60
    assert len({cell.key for cell in cells}) == 60
    assert sum(cell.configuration == "optimized" for cell in cells) == 36
    assert sum(cell.configuration == "cpu_baseline" for cell in cells) == 24
    assert any(
        cell.method == "bm"
        and cell.dimension == 200
        and cell.implementation == "pybspcov"
        and cell.device == "gpu"
        and cell.dtype == "float32"
        and cell.parallelism == 8
        and cell.cpu_cores == 8
        for cell in cells
    )
    assert all(
        cell.dtype == "float64"
        for cell in cells
        if cell.configuration == "cpu_baseline"
    )


def test_matrix_uses_available_affinity_and_enables_python_x64(tmp_path: Path) -> None:
    matrix = _module(MATRIX_PATH, "r_comparison_matrix_affinity")
    manifest = _core().load_manifest(MANIFEST_PATH)
    cell = next(
        cell
        for cell in matrix.build_cells(manifest)
        if cell.implementation == "pybspcov"
        and cell.configuration == "optimized"
        and cell.dtype == "float64"
    )

    command = matrix.command_for_cell(
        cell,
        script_directory=BENCHMARK_DIR,
        manifest_path=MANIFEST_PATH,
        fixture_root=tmp_path / "fixtures",
        output_directory=tmp_path / "output",
        available_cpu_ids=(4, 6, 8, 10, 12, 14, 16, 18),
    )
    environment = matrix.benchmark_environment({"JAX_ENABLE_X64": "0"})

    assert command[:3] == ["taskset", "-c", "4,6,8,10,12,14,16,18"]
    assert environment["JAX_ENABLE_X64"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_matrix_replaces_runner_cold_time_with_external_process_time(
    tmp_path: Path,
) -> None:
    matrix = _module(MATRIX_PATH, "r_comparison_matrix_cold")
    output = tmp_path / "cell.jsonl"
    output.write_text(
        json.dumps({"cold_end_to_end_seconds": 1.0, "other": "preserved"}) + "\n",
        encoding="utf-8",
    )

    matrix.record_external_cold_time(output, 2.5)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "cold_end_to_end_seconds": 2.5,
        "other": "preserved",
    }


def test_r_runner_help_exposes_all_four_methods_without_loading_bspcov() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is unavailable")
    assert R_RUNNER_PATH.is_file(), "benchmarks/r_comparison/run_bspcov.R is required"

    result = subprocess.run(
        ["Rscript", "--vanilla", str(R_RUNNER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "bm|sbm|bandppp|thresholdppp" in result.stdout
    assert "bspcov 1.0.3" in result.stdout


def test_r_parity_runner_help_does_not_load_bspcov() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is unavailable")
    assert R_PARITY_RUNNER_PATH.is_file()

    result = subprocess.run(
        ["Rscript", "--vanilla", str(R_PARITY_RUNNER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "bm|sbm|bandppp|thresholdppp" in result.stdout
    assert "bspcov 1.0.3" in result.stdout


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("bm", {"burnin": 2000, "n_samples": 2000, "n_chains": 4}),
        (
            "sbm",
            {
                "burnin": 2000,
                "n_samples": 2000,
                "n_chains": 4,
                "cutoff_method": "correlation",
                "screening_scope": "chain",
            },
        ),
        (
            "bandppp",
            {
                "bandwidth": 1,
                "epsilon": 0.05,
                "n_samples": 5000,
                "n_chains": 1,
            },
        ),
        (
            "thresholdppp",
            {
                "threshold": 0.1,
                "method": "hard",
                "epsilon": 0.1,
                "n_samples": 5000,
                "n_chains": 1,
            },
        ),
    ],
)
def test_parity_runner_uses_long_pre_registered_configuration(
    method: str, expected: dict[str, object]
) -> None:
    runner = _module(PARITY_RUNNER_PATH, "r_comparison_parity_runner")
    manifest = _core().load_manifest(MANIFEST_PATH)

    estimator = runner.build_parity_estimator(
        method, dtype="float64", device="cpu", manifest=manifest
    )

    for attribute, value in expected.items():
        assert getattr(estimator, attribute) == value


def _scalar_summary(mean: float, *, mcse: float = 0.01) -> dict[str, object]:
    return {
        "truth": [[1.0]],
        "posterior_mean": [[mean]],
        "posterior_mean_mcse": [[mcse]],
        "posterior_sd": [[0.2]],
        "posterior_sd_mcse": [[mcse]],
        "q025": [[0.7]],
        "q025_mcse": [[mcse]],
        "q50": [[1.0]],
        "q50_mcse": [[mcse]],
        "q975": [[1.3]],
        "q975_mcse": [[mcse]],
        "rmse": 0.0,
        "rmse_mcse": mcse,
    }


def test_parity_artifacts_require_matching_fixture_and_preserve_detailed_verdicts() -> (
    None
):
    comparison = _module(PARITY_COMPARE_PATH, "r_comparison_parity_compare")
    r_artifact = {
        "method": "bm",
        "implementation": "bspcov",
        "version": "1.0.3",
        "dtype": "float64",
        "fixture_sha256": "a" * 64,
        "git_revision": "c" * 40,
        "git_dirty": False,
        "summary": _scalar_summary(1.0),
    }
    python64 = {
        "method": "bm",
        "implementation": "pybspcov",
        "version": "0.1.0.dev0",
        "dtype": "float64",
        "fixture_sha256": "a" * 64,
        "git_revision": "c" * 40,
        "git_dirty": False,
        "summary": _scalar_summary(1.01),
    }
    python32 = {
        **python64,
        "dtype": "float32",
        "summary": _scalar_summary(1.2),
    }

    result = comparison.compare_method_artifacts(r_artifact, [python64, python32])

    assert result["float64"]["verdict"] == "pass"
    assert result["float32"]["verdict"] == "fail"
    assert (
        result["float32"]["comparison"]["statistics"]["posterior_mean"][0][0][
            "within_tolerance"
        ]
        is False
    )

    python32["fixture_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="fixture"):
        comparison.compare_method_artifacts(r_artifact, [python64, python32])

    python32["fixture_sha256"] = "a" * 64
    python32["git_revision"] = "d" * 40
    with pytest.raises(ValueError, match="revision"):
        comparison.compare_method_artifacts(r_artifact, [python64, python32])
