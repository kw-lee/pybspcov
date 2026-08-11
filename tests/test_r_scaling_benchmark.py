from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

BENCHMARK_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "r_scaling" / "run_p200.py"
)


def _benchmark_module() -> ModuleType:
    assert BENCHMARK_PATH.exists(), "benchmarks/r_scaling/run_p200.py is required"
    spec = importlib.util.spec_from_file_location("run_p200", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalized_timing_matches_parallel_and_sequential_per_chain_work() -> None:
    benchmark = _benchmark_module()

    parallel = benchmark.normalize_chain_timing(
        [40.0], execution_model="parallel", chain_count=4
    )
    sequential = benchmark.normalize_chain_timing(
        [8.0, 10.0, 12.0, 10.0],
        execution_model="sequential",
        chain_count=4,
    )

    assert parallel == {
        "execution_model": "parallel",
        "raw_wall_seconds": [40.0],
        "total_wall_seconds": 40.0,
        "normalized_wall_seconds_per_chain": 10.0,
        "chains_per_second": 0.1,
    }
    assert sequential == {
        "execution_model": "sequential",
        "raw_wall_seconds": [8.0, 10.0, 12.0, 10.0],
        "total_wall_seconds": 40.0,
        "normalized_wall_seconds_per_chain": 10.0,
        "chains_per_second": 0.1,
    }


class FakeEstimator:
    def __init__(self, configurations: list[dict[str, object]], **kwargs: object):
        configurations.append(dict(kwargs))
        self.n_chains = int(kwargs["n_chains"])
        self.n_samples = int(kwargs["n_samples"])

    def fit(
        self,
        observations: np.ndarray,
        *,
        key: Any,
        initial_covariance: np.ndarray,
    ) -> FakeEstimator:
        del observations, key, initial_covariance
        identity = np.eye(2, dtype=np.float64)
        self.posterior_samples_ = np.broadcast_to(
            identity,
            (self.n_chains, self.n_samples, 2, 2),
        ).copy()
        return self


def test_cpu_mode_normalizes_one_vmapped_four_chain_fit() -> None:
    benchmark = _benchmark_module()
    configurations: list[dict[str, object]] = []
    clock_values = iter([0.0, 5.0, 10.0, 30.0])

    result = benchmark.measure_python_mode(
        np.zeros((6, 2), dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        device="cpu",
        dtype="float64",
        burnin=3,
        n_samples=2,
        chain_count=4,
        seed=11,
        estimator_factory=lambda **kwargs: FakeEstimator(configurations, **kwargs),
        clock=lambda: next(clock_values),
    )

    assert [configuration["n_chains"] for configuration in configurations] == [4, 4]
    assert result["compile_plus_execution_seconds"] == 5.0
    assert result["normalized_wall_seconds_per_chain"] == 5.0
    assert result["chains_per_second"] == 0.2
    assert result["retained_draws"] == 8
    assert result["posterior_mean_spd"] is True


def test_gpu_mode_averages_four_sequential_single_chain_fits() -> None:
    benchmark = _benchmark_module()
    configurations: list[dict[str, object]] = []
    clock_values = iter([0.0, 5.0, 10.0, 18.0, 20.0, 30.0, 40.0, 52.0, 60.0, 70.0])

    result = benchmark.measure_python_mode(
        np.zeros((6, 2), dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        device="gpu",
        dtype="float64",
        burnin=3,
        n_samples=2,
        chain_count=4,
        seed=11,
        estimator_factory=lambda **kwargs: FakeEstimator(configurations, **kwargs),
        clock=lambda: next(clock_values),
    )

    assert [configuration["n_chains"] for configuration in configurations] == [
        1,
        1,
        1,
        1,
        1,
    ]
    assert result["compile_plus_execution_seconds"] == 5.0
    assert result["normalized_wall_seconds_per_chain"] == 10.0
    assert result["chains_per_second"] == 0.1
    assert result["retained_draws"] == 8
    assert result["posterior_mean_spd"] is True


def test_shared_fixture_export_preserves_hash_and_initial_variances(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark_module()

    manifest = benchmark.write_shared_fixture(
        tmp_path,
        dimension=4,
        density=0.25,
        n_observations=12,
        seed=17,
    )

    observations = np.loadtxt(tmp_path / "observations.csv", delimiter=",")
    truth = np.loadtxt(tmp_path / "truth_covariance.csv", delimiter=",")
    initial = np.loadtxt(tmp_path / "initial_covariance.csv", delimiter=",")
    assert observations.shape == (12, 4)
    assert truth.shape == (4, 4)
    assert initial.shape == (4, 4)
    assert np.allclose(observations.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(initial, np.diag(np.var(observations, axis=0, ddof=1)))
    assert manifest == {
        "dimension": 4,
        "n_observations": 12,
        "density": 0.25,
        "seed": 17,
        "dtype": "float64",
        "fixture_sha256": manifest["fixture_sha256"],
    }
    assert len(manifest["fixture_sha256"]) == 64
