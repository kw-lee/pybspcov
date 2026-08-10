import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "r_example"
sys.path.insert(0, str(BENCHMARK_DIR))

from run_pybspcov import summarize_draws

from pybspcov import SBMSPCov
from pybspcov.kernels import unpack_lower_triangle_column_major

DATA_DIR = BENCHMARK_DIR / "data"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"
SUMMARY_PATH = FIXTURE_DIR / "sbm_public_corr_summary.csv"
METADATA_PATH = FIXTURE_DIR / "sbm_public_corr_metadata.json"
STATISTICS = ("posterior_mean", "posterior_sd", "q025", "q50", "q975")
EXPECTED_SCREENING_MASK = np.array(
    [
        [False, False, False, True, False],
        [False, False, False, False, False],
        [False, False, False, True, False],
        [True, False, True, False, False],
        [False, False, False, False, False],
    ]
)
EXPECTED_METADATA: dict[str, Any] = {
    "fixture_schema_version": 1,
    "implementation": "bspcov",
    "package": "bspcov",
    "package_version": "1.0.3",
    "runtime": "R",
    "runtime_version": "4.5.0",
    "model": "sbm",
    "sampler_variant": "bspcov_sbm",
    "estimator_api": "bspcov::sbmspcov",
    "dtype": "float64",
    "fixture_sha256": (
        "1da3b680d62bb9bd1b5d9fbc26309ce4be7c5e58f51d3d4b84ce76d132dea22d"
    ),
    "initial_fixture_sha256": (
        "9e2fb258876e884486abcb66855cd2d000bda03ef892df377038f15d11627ab2"
    ),
    "n": 20,
    "p": 5,
    "burnin": 1000,
    "n_samples": 1000,
    "chains": 1,
    "sampler_seed": 1,
    "screening_method": "corr",
    "screening_retained_fraction": 0.2,
    "screening_active_lower": "0010000100",
    "screening_active_edges": 2,
    "prior_a": 0.5,
    "prior_b": 0.5,
    "diagonal_rate": 1.0,
    "n_batches": 20,
    "batch_size": 50,
    "trimmed_samples": 1000,
}


def _read_r_summary(dimension: int) -> dict[str, Any]:
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    expected_keys = {
        (row, column)
        for row in range(1, dimension + 1)
        for column in range(1, dimension + 1)
    }
    keyed_rows = {(int(row["row"]), int(row["column"])): row for row in rows}
    assert keyed_rows.keys() == expected_keys
    assert {row["implementation"] for row in rows} == {"bspcov"}

    summary: dict[str, Any] = {}
    for statistic in STATISTICS:
        summary[statistic] = np.array(
            [
                [float(keyed_rows[(row, column)][statistic]) for column in range(1, 6)]
                for row in range(1, 6)
            ]
        )
        mcse_field = f"{statistic}_mcse"
        summary[mcse_field] = np.array(
            [
                [float(keyed_rows[(row, column)][mcse_field]) for column in range(1, 6)]
                for row in range(1, 6)
            ]
        )

    first_row = keyed_rows[(1, 1)]
    summary["rmse"] = float(first_row["rmse"])
    summary["rmse_mcse"] = float(first_row["rmse_mcse"])
    assert {float(row["rmse"]) for row in rows} == {summary["rmse"]}
    assert {float(row["rmse_mcse"]) for row in rows} == {summary["rmse_mcse"]}
    return summary


def _assert_six_combined_mcse(
    r_summary: dict[str, Any],
    pybspcov_summary: dict[str, Any],
) -> None:
    failures = []
    for statistic in (*STATISTICS, "rmse"):
        difference = np.abs(pybspcov_summary[statistic] - r_summary[statistic])
        combined_mcse = np.hypot(
            pybspcov_summary[f"{statistic}_mcse"],
            r_summary[f"{statistic}_mcse"],
        )
        tolerance = 6.0 * combined_mcse
        if not bool(np.all(difference <= tolerance)):
            failures.append(
                f"{statistic}: max difference={float(np.max(difference)):.17g}, "
                f"max tolerance={float(np.max(tolerance)):.17g}"
            )
    assert not failures, "posterior parity exceeded six combined MCSEs:\n" + "\n".join(
        failures
    )


def test_public_sbm_posterior_matches_bspcov_1_0_3_golden() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    r_summary = _read_r_summary(dimension=5)
    assert {field: metadata[field] for field in EXPECTED_METADATA} == EXPECTED_METADATA
    assert metadata["screening_cutoff"] == 0.25561133439277933
    assert metadata["tau1sq"] == math.log(5) / (5**2 * 20)
    assert metadata["source_command"] == (
        "Rscript --vanilla reference/r/generate_sbm_public_corr_fixture.R"
    )
    assert metadata["r_platform"]
    assert metadata["blas"]
    assert metadata["lapack"]
    assert metadata["session_info"]

    x = np.loadtxt(DATA_DIR / "bm_example_x.csv", delimiter=",")
    truth = np.loadtxt(DATA_DIR / "bm_example_truth.csv", delimiter=",")
    initial_covariance = np.loadtxt(DATA_DIR / "bm_example_initial.csv", delimiter=",")
    model = SBMSPCov(
        n_samples=1000,
        burnin=1000,
        n_chains=1,
        cutoff_method="correlation",
        retained_fraction=0.2,
        dtype="float64",
        device="cpu",
    )

    model.fit(
        x,
        key=jax.random.key(1),
        initial_covariance=initial_covariance,
    )

    assert model.dtype_ == jnp.dtype("float64")
    assert model.device_.platform == "cpu"
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(model.screening_mask_)),
        EXPECTED_SCREENING_MASK,
    )
    assert model.diagnostics_.n_active_edges == 2
    packed_draws = model.posterior_samples_packed_[0]
    assert packed_draws.shape == (1000, 15)
    draws = unpack_lower_triangle_column_major(packed_draws, dimension=5)
    pybspcov_summary = summarize_draws(
        np.asarray(jax.device_get(draws)),
        truth,
    )

    _assert_six_combined_mcse(r_summary, pybspcov_summary)
