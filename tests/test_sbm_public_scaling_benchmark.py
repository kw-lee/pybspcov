from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "sbm_public_scaling.py"


def _benchmark_module() -> ModuleType:
    assert BENCHMARK_PATH.exists(), "benchmarks/sbm_public_scaling.py is required"
    spec = importlib.util.spec_from_file_location("sbm_public_scaling", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_sha256(
    precision: np.ndarray,
    covariance: np.ndarray,
    observations: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("precision", precision),
        ("covariance", covariance),
        ("observations", observations),
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def test_fixture_generator_is_deterministic_sparse_centered_and_spd() -> None:
    benchmark = _benchmark_module()

    first = benchmark.generate_fixture(
        dimension=8,
        density=0.25,
        n_observations=24,
        seed=20260803,
        dtype="float64",
    )
    second = benchmark.generate_fixture(
        dimension=8,
        density=0.25,
        n_observations=24,
        seed=20260803,
        dtype="float64",
    )
    changed = benchmark.generate_fixture(
        dimension=8,
        density=0.25,
        n_observations=24,
        seed=20260804,
        dtype="float64",
    )

    precision = np.asarray(first.precision)
    covariance = np.asarray(first.covariance)
    observations = np.asarray(first.observations)
    assert precision.shape == (8, 8)
    assert covariance.shape == (8, 8)
    assert observations.shape == (24, 8)
    assert np.array_equal(precision, np.asarray(second.precision))
    assert np.array_equal(covariance, np.asarray(second.covariance))
    assert np.array_equal(observations, np.asarray(second.observations))
    assert first.sha256 == second.sha256
    assert changed.sha256 != first.sha256
    assert np.array_equal(precision, precision.T)
    off_diagonal = precision.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.count_nonzero(np.tril(off_diagonal, k=-1)) == round(0.25 * 28)
    assert np.all(np.diag(precision) > np.sum(np.abs(off_diagonal), axis=1))
    assert np.all(np.linalg.eigvalsh(precision) > 0.0)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert np.allclose(covariance @ precision, np.eye(8), rtol=1e-11, atol=1e-11)
    assert np.allclose(observations.mean(axis=0), 0.0, rtol=0.0, atol=1e-12)
    assert first.sha256 == _fixture_sha256(precision, covariance, observations)


class BlockingArray:
    def __init__(self, value: np.ndarray, label: str, blocked: list[str]) -> None:
        self.value = value
        self.label = label
        self.blocked = blocked

    def block_until_ready(self) -> BlockingArray:
        self.blocked.append(self.label)
        return self

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        value = np.asarray(self.value, dtype=dtype)
        return value.copy() if copy else value


class FakeEstimator:
    def __init__(
        self,
        *,
        index: int,
        fit_keys: list[bytes],
        blocked: list[str],
    ) -> None:
        self.index = index
        self.fit_keys = fit_keys
        self.blocked = blocked

    def fit(
        self,
        observations: np.ndarray,
        *,
        key: jax.Array,
    ) -> FakeEstimator:
        assert observations.shape == (6, 2)
        self.fit_keys.append(np.asarray(jax.random.key_data(key)).tobytes())
        prefix = f"fit-{self.index}"
        self.covariance_ = BlockingArray(
            np.asarray([[2.2, 0.0], [0.0, 2.7]], dtype=np.float32),
            f"{prefix}:covariance",
            self.blocked,
        )
        self.posterior_samples_packed_ = BlockingArray(
            np.ones((1, 2, 3), dtype=np.float32),
            f"{prefix}:posterior",
            self.blocked,
        )
        self.phi_samples_packed_ = BlockingArray(
            np.ones((1, 2, 3), dtype=np.float32),
            f"{prefix}:phi",
            self.blocked,
        )
        self.screening_mask_ = BlockingArray(
            np.asarray([[False, True], [True, False]]),
            f"{prefix}:screening-mask",
            self.blocked,
        )
        accepted = BlockingArray(
            np.ones((1, 4), dtype=np.bool_),
            f"{prefix}:accepted",
            self.blocked,
        )
        self.diagnostics_ = SimpleNamespace(
            accepted=accepted,
            n_active_edges=1,
            n_rejected_sweeps=0,
        )
        self.dtype_ = np.dtype("float32")
        return self


def test_public_runner_uses_fresh_estimators_keys_and_synchronizes_outputs() -> None:
    benchmark = _benchmark_module()
    fit_keys: list[bytes] = []
    blocked: list[str] = []
    configurations: list[dict[str, object]] = []
    estimators: list[FakeEstimator] = []

    def estimator_factory(**configuration: object) -> FakeEstimator:
        configurations.append(configuration)
        estimator = FakeEstimator(
            index=len(estimators),
            fit_keys=fit_keys,
            blocked=blocked,
        )
        estimators.append(estimator)
        return estimator

    clock_values = iter([0.0, 5.0, 10.0, 12.0, 20.0, 23.0, 30.0, 34.0])
    memory_phases: list[str] = []

    def memory_probe(phase: str) -> int:
        memory_phases.append(phase)
        return {"before": 100, "after": 140, "peak": 180}[phase]

    summary = benchmark.run_public_fit_benchmark(
        np.asarray(
            [
                [-1.0, 0.0],
                [-0.6, 0.4],
                [-0.2, -0.4],
                [0.2, 0.2],
                [0.6, -0.2],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        estimator_factory=estimator_factory,
        estimator_kwargs={
            "burnin": 2,
            "n_samples": 2,
            "n_chains": 1,
            "dtype": "float32",
            "device": "cpu",
        },
        repetitions=3,
        seed=41,
        clock=lambda: next(clock_values),
        memory_probe=memory_probe,
    )

    assert len(estimators) == 4
    assert (
        configurations
        == [
            {
                "burnin": 2,
                "n_samples": 2,
                "n_chains": 1,
                "dtype": "float32",
                "device": "cpu",
            }
        ]
        * 4
    )
    assert len(fit_keys) == 4
    assert len(set(fit_keys)) == 4
    assert blocked == [
        f"fit-{index}:{leaf}"
        for index in range(4)
        for leaf in ("covariance", "posterior", "phi", "screening-mask", "accepted")
    ]
    assert summary["first_fit_seconds"] == 5.0
    assert summary["warmed_fit_seconds"] == {
        "raw": [2.0, 3.0, 4.0],
        "median": 3.0,
        "min": 2.0,
        "max": 4.0,
    }
    assert summary["posterior_mean_finite"] is True
    assert summary["posterior_mean_symmetric"] is True
    assert summary["posterior_mean_spd"] is True
    assert summary["dtype"] == "float32"
    assert summary["accepted_sweeps"] == 4
    assert summary["rejected_sweeps"] == 0
    assert summary["active_edges"] == 1
    assert summary["compact_width"] == 1
    expected_error = np.linalg.norm(np.diag([0.2, -0.3])) / np.linalg.norm(
        np.diag([2.0, 3.0])
    )
    assert summary["truth_relative_frobenius_error"] == pytest.approx(expected_error)
    assert memory_phases == ["before", "after", "peak"]
    assert summary["device_memory_bytes"] == {
        "before": 100,
        "after": 140,
        "peak": 180,
    }


def test_public_runner_uses_only_the_estimator_fit_seam() -> None:
    benchmark = _benchmark_module()

    class FitOnlyEstimator:
        def fit(
            self,
            observations: np.ndarray,
            *,
            key: jax.Array,
        ) -> FitOnlyEstimator:
            del observations, key
            raise RuntimeError("public fit seam reached")

    with pytest.raises(RuntimeError, match="public fit seam reached"):
        benchmark.run_public_fit_benchmark(
            np.zeros((4, 2), dtype=np.float32),
            np.eye(2, dtype=np.float32),
            estimator_factory=lambda **_: FitOnlyEstimator(),
            estimator_kwargs={},
            repetitions=1,
            seed=7,
            clock=lambda: 0.0,
            memory_probe=None,
        )


def test_cli_emits_one_provenance_record_per_requested_dimension(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module()
    fixture_calls: list[tuple[int, float, int, int, str]] = []
    runner_calls: list[tuple[int, int]] = []

    def fake_fixture(
        *,
        dimension: int,
        density: float,
        n_observations: int,
        seed: int,
        dtype: str,
    ) -> SimpleNamespace:
        fixture_calls.append((dimension, density, n_observations, seed, dtype))
        return SimpleNamespace(
            observations=np.zeros((n_observations, dimension), dtype=np.float32),
            covariance=np.eye(dimension, dtype=np.float32),
            precision=np.eye(dimension, dtype=np.float32),
            sha256=f"{dimension:064x}",
        )

    def fake_runner(
        observations: np.ndarray,
        truth_covariance: np.ndarray,
        **kwargs: object,
    ) -> dict[str, object]:
        del truth_covariance
        runner_seed = kwargs["seed"]
        assert isinstance(runner_seed, int)
        runner_calls.append((observations.shape[1], runner_seed))
        return {
            "first_fit_seconds": 1.0,
            "warmed_fit_seconds": {
                "raw": [0.5, 0.6],
                "median": statistics.median([0.5, 0.6]),
                "min": 0.5,
                "max": 0.6,
            },
            "posterior_mean_finite": True,
            "posterior_mean_symmetric": True,
            "posterior_mean_spd": True,
            "dtype": "float32",
            "accepted_sweeps": 3,
            "rejected_sweeps": 0,
            "active_edges": 2,
            "compact_width": 1,
            "truth_relative_frobenius_error": 0.1,
            "device_memory_bytes": None,
        }

    monkeypatch.setattr(benchmark, "generate_fixture", fake_fixture)
    monkeypatch.setattr(benchmark, "run_public_fit_benchmark", fake_runner)
    monkeypatch.setattr(
        benchmark,
        "_git_provenance",
        lambda _: {"revision": "abc123", "dirty": False},
    )
    monkeypatch.setattr(
        benchmark,
        "_environment_metadata",
        lambda: {"python": "3.12", "jax": "0.11"},
    )

    benchmark.main(
        [
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--dimensions",
            "25",
            "50",
            "100",
            "200",
            "--density",
            "0.05",
            "--n-factor",
            "3",
            "--burnin",
            "1",
            "--samples",
            "2",
            "--repetitions",
            "2",
            "--seed",
            "19",
        ]
    )

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["dimension"] for record in records] == [25, 50, 100, 200]
    assert fixture_calls == [
        (dimension, 0.05, dimension * 3, 19, "float32")
        for dimension in (25, 50, 100, 200)
    ]
    assert runner_calls == [(25, 19), (50, 19), (100, 19), (200, 19)]
    for record in records:
        assert record.keys() >= {
            "accepted_sweeps",
            "active_edges",
            "benchmark",
            "burnin",
            "compact_width",
            "density",
            "device",
            "device_memory_bytes",
            "dimension",
            "dtype",
            "environment",
            "first_fit_seconds",
            "fixture_sha256",
            "git",
            "n_observations",
            "posterior_mean_finite",
            "posterior_mean_spd",
            "posterior_mean_symmetric",
            "rejected_sweeps",
            "repetitions",
            "samples",
            "schema_version",
            "seed",
            "truth_relative_frobenius_error",
            "warmed_fit_seconds",
        }
        assert record["benchmark"] == "sbm-public-scaling"
        assert record["schema_version"] == "1.0"
        assert record["device"] == "cpu"
        assert record["dtype"] == "float32"
        assert record["density"] == 0.05
        assert record["n_observations"] == record["dimension"] * 3
        assert record["burnin"] == 1
        assert record["samples"] == 2
        assert record["repetitions"] == 2
        assert record["seed"] == 19
        assert record["fixture_sha256"] == f"{record['dimension']:064x}"
        assert record["git"] == {"revision": "abc123", "dirty": False}
        assert record["environment"] == {"python": "3.12", "jax": "0.11"}
