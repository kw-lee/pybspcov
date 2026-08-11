from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "r_scaling" / "run_p200.py"


def _benchmark_module() -> ModuleType:
    assert BENCHMARK_PATH.exists(), "benchmarks/r_scaling/run_p200.py is required"
    spec = importlib.util.spec_from_file_location("run_p200", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _r_repetitions() -> list[dict[str, object]]:
    return [
        {
            "repetition": index,
            "seed": 100 + index,
            "raw_wall_seconds": [4.0 * seconds],
            "total_wall_seconds": 4.0 * seconds,
            "normalized_wall_seconds_per_chain": seconds,
            "chains_per_second": 1.0 / seconds,
            "retained_draws": 200,
            "posterior_mean_finite": True,
            "posterior_mean_symmetric": True,
            "posterior_mean_spd": True,
            "truth_relative_frobenius_error": 0.18 + index / 1000.0,
        }
        for index, seconds in enumerate(np.arange(1.0, 11.0))
    ]


def test_build_r_result_records_ten_repetitions_without_python_results() -> None:
    """Catch reintroducing duplicate Python execution or single-run R output."""
    benchmark = _benchmark_module()
    fixture = {
        "dimension": 200,
        "n_observations": 600,
        "density": 0.05,
        "seed": 20260803,
        "dtype": "float64",
        "fixture_sha256": "a" * 64,
    }

    result = benchmark.build_r_result(
        _r_repetitions(),
        fixture=fixture,
        burnin=50,
        n_samples=50,
        chain_count=4,
        r_metadata={"package_version": "1.0.3"},
    )

    assert "python" not in result
    assert result["schema_version"] == "2.0"
    assert result["configuration"] == {
        "burnin": 50,
        "n_samples": 50,
        "chain_count": 4,
        "repetitions": 10,
    }
    assert len(result["r"]["measured_repetitions"]) == 10
    assert result["r"]["timing_summary"] == {
        "median": 5.5,
        "q1": 3.25,
        "q3": 7.75,
        "min": 1.0,
        "max": 10.0,
    }
    assert result["r"]["package_version"] == "1.0.3"


def test_build_r_result_rejects_incomplete_or_invalid_repetitions() -> None:
    """Catch publishing an R run with missing draws or invalid covariance."""
    benchmark = _benchmark_module()
    records = _r_repetitions()
    records[3] = {**records[3], "retained_draws": 199}

    with pytest.raises(ValueError, match="200 retained draws"):
        benchmark.build_r_result(
            records,
            fixture={"fixture_sha256": "a" * 64},
            burnin=50,
            n_samples=50,
            chain_count=4,
        )


def test_r_invocation_propagates_ten_repetitions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch forgetting the repeat count at the Python-to-R boundary."""
    benchmark = _benchmark_module()
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **_: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "metadata": {"package_version": "1.0.3"},
                    "measured_repetitions": [],
                }
            )
            + "\n"
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    benchmark._run_r(
        tmp_path,
        r_library=tmp_path / "r-library",
        fixture_sha256="a" * 64,
        burnin=50,
        n_samples=50,
        chain_count=4,
        repetitions=10,
        seed=20260803,
    )

    assert len(commands) == 1
    repeat_option = commands[0].index("--repetitions")
    assert commands[0][repeat_option + 1] == "10"


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
