import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "r_example"
PYBSPCOV_RUNNER = BENCHMARK_DIR / "run_pybspcov.py"
COMPARISON_RUNNER = BENCHMARK_DIR / "compare_results.py"
R_RUNNER = BENCHMARK_DIR / "run_bspcov.R"
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


def _run_pybspcov(
    output_directory: Path, device: str
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = device
    if device == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [
            sys.executable,
            str(PYBSPCOV_RUNNER),
            "--burnin",
            "0",
            "--n-samples",
            "4",
            "--repetitions",
            "1",
            "--device",
            device,
            "--output-dir",
            str(output_directory),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_pybspcov_cpu_cli_writes_all_result_schemas(tmp_path: Path) -> None:
    result = _run_pybspcov(tmp_path, "cpu")

    assert result.returncode == 0, result.stderr
    with (tmp_path / "pybspcov_summary.csv").open(newline="", encoding="utf-8") as file:
        summary_reader = csv.DictReader(file)
        summary_rows = list(summary_reader)
    assert tuple(summary_reader.fieldnames or ()) == SUMMARY_FIELDS
    assert len(summary_rows) == 25
    with (tmp_path / "pybspcov_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        metadata_rows = list(csv.DictReader(file))
    assert {row["name"] for row in metadata_rows} >= {
        "package_version",
        "dtype",
        "device_kind",
        "n_batches",
        "steady_state_timing_scope",
    }
    with (tmp_path / "pybspcov_timing.csv").open(newline="", encoding="utf-8") as file:
        timing_reader = csv.DictReader(file)
        timing_rows = list(timing_reader)
    assert tuple(timing_reader.fieldnames or ()) == (
        "implementation",
        "compile_plus_execution_seconds",
        "steady_state_seconds",
        "steady_state_min_seconds",
        "steady_state_max_seconds",
        "end_to_end_seconds",
    )
    assert len(timing_rows) == 1


def test_pybspcov_gpu_cli_fails_clearly_when_cuda_is_hidden(tmp_path: Path) -> None:
    result = _run_pybspcov(tmp_path, "gpu")

    assert result.returncode == 2
    assert "requested JAX GPU device is unavailable" in result.stderr
    assert not list(tmp_path.glob("*.csv"))


def _write_csv(path: Path, fieldnames: tuple[str, ...], row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def test_comparison_cli_writes_json_verdict(tmp_path: Path) -> None:
    summary_row: dict[str, object] = {
        "implementation": "placeholder",
        "row": 1,
        "column": 1,
        "posterior_mean": 1.0,
        "posterior_mean_mcse": 0.1,
        "posterior_sd": 0.5,
        "posterior_sd_mcse": 0.1,
        "q025": 0.1,
        "q025_mcse": 0.1,
        "q50": 1.0,
        "q50_mcse": 0.1,
        "q975": 1.9,
        "q975_mcse": 0.1,
        "truth": 1.0,
        "rmse": 0.0,
        "rmse_mcse": 0.1,
    }
    r_summary = tmp_path / "r_summary.csv"
    pybspcov_summary = tmp_path / "pybspcov_summary.csv"
    _write_csv(r_summary, SUMMARY_FIELDS, summary_row | {"implementation": "bspcov"})
    _write_csv(
        pybspcov_summary,
        SUMMARY_FIELDS,
        summary_row | {"implementation": "pybspcov"},
    )
    r_timing = tmp_path / "r_timing.csv"
    pybspcov_timing = tmp_path / "pybspcov_timing.csv"
    _write_csv(
        r_timing,
        ("implementation", "sampler_seconds", "end_to_end_seconds"),
        {"implementation": "bspcov", "sampler_seconds": 2.0, "end_to_end_seconds": 3.0},
    )
    _write_csv(
        pybspcov_timing,
        (
            "implementation",
            "compile_plus_execution_seconds",
            "steady_state_seconds",
            "end_to_end_seconds",
        ),
        {
            "implementation": "pybspcov",
            "compile_plus_execution_seconds": 2.0,
            "steady_state_seconds": 0.5,
            "end_to_end_seconds": 3.0,
        },
    )
    output = tmp_path / "comparison.json"

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARISON_RUNNER),
            "--r-summary",
            str(r_summary),
            "--pybspcov-summary",
            str(pybspcov_summary),
            "--r-timing",
            str(r_timing),
            "--pybspcov-timing",
            str(pybspcov_timing),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["statistical_verdict"] == "pass"
    assert payload["timing_categories"]["pybspcov"]["steady_state_seconds"] == 0.5


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
@pytest.mark.parametrize(
    ("draw_setup", "message"),
    [
        ("draws <- matrix(1, nrow = 3, ncol = 3)", "returned 3 draws; expected 4"),
        ("draws <- matrix(1, nrow = 4, ncol = 2)", "returned 2 columns; expected 3"),
        (
            "draws <- matrix(1, nrow = 4, ncol = 3); draws[1, 1] <- NaN",
            "nonfinite",
        ),
    ],
)
def test_r_draw_validation_rejects_invalid_sampler_output(
    draw_setup: str, message: str
) -> None:
    expression = (
        "expressions <- parse(file = commandArgs(TRUE)[1]); "
        "names <- vapply(expressions, function(value) "
        "if (is.call(value) && identical(value[[1]], as.name('<-'))) "
        "as.character(value[[2]]) else '', character(1)); "
        "eval(expressions[[which(names == 'validate_draws')[[1]]]]); "
        f"{draw_setup}; validate_draws(draws, 4L, 2L)"
    )

    result = subprocess.run(
        ["Rscript", "-e", expression, str(R_RUNNER)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
