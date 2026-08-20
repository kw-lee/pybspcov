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


def test_repeated_runner_uses_one_vmapped_four_chain_fit_per_cpu_run() -> None:
    """Catch accidentally timing single-chain or sequential CPU fits."""
    benchmark = _benchmark_module()
    configurations: list[dict[str, object]] = []
    estimators: list[FakeEstimator] = []
    fit_keys: list[bytes] = []
    blocked: list[str] = []

    def estimator_factory(**configuration: object) -> FakeEstimator:
        configurations.append(dict(configuration))
        estimator = FakeEstimator(
            index=len(estimators),
            fit_keys=fit_keys,
            blocked=blocked,
        )
        estimators.append(estimator)
        return estimator

    clock_values = iter([0.0, 5.0, 10.0, 30.0, 40.0, 64.0])
    result = benchmark.run_repeated_fit_benchmark(
        np.zeros((6, 2), dtype=np.float32),
        np.diag([2.0, 3.0]).astype(np.float32),
        estimator_factory=estimator_factory,
        estimator_kwargs={
            "burnin": 2,
            "n_samples": 2,
            "dtype": "float32",
            "device": "cpu",
        },
        execution_model="parallel",
        chain_count=4,
        repetitions=2,
        seed=41,
        clock=lambda: next(clock_values),
    )

    assert [configuration["n_chains"] for configuration in configurations] == [
        4,
        4,
        4,
    ]
    assert result["compile_plus_execution_seconds"] == 5.0
    assert [
        repetition["normalized_wall_seconds_per_chain"]
        for repetition in result["measured_repetitions"]
    ] == [5.0, 6.0]
    assert all(
        repetition["retained_draws"] == 8
        for repetition in result["measured_repetitions"]
    )
    assert len(set(fit_keys)) == 3
    assert blocked == [
        f"fit-{index}:{leaf}"
        for index in range(3)
        for leaf in ("covariance", "posterior", "phi", "screening-mask", "accepted")
    ]


def test_repeated_runner_uses_four_sequential_single_chain_gpu_fits() -> None:
    """Catch failing to sum the intentionally sequential GPU chain timings."""
    benchmark = _benchmark_module()
    configurations: list[dict[str, object]] = []
    estimators: list[FakeEstimator] = []

    def estimator_factory(**configuration: object) -> FakeEstimator:
        configurations.append(dict(configuration))
        estimator = FakeEstimator(index=len(estimators), fit_keys=[], blocked=[])
        estimators.append(estimator)
        return estimator

    clock_values = iter(float(value) for value in range(18))
    result = benchmark.run_repeated_fit_benchmark(
        np.zeros((6, 2), dtype=np.float32),
        np.diag([2.0, 3.0]).astype(np.float32),
        estimator_factory=estimator_factory,
        estimator_kwargs={
            "burnin": 2,
            "n_samples": 2,
            "dtype": "float32",
            "device": "gpu",
        },
        execution_model="sequential",
        chain_count=4,
        repetitions=2,
        seed=41,
        clock=lambda: next(clock_values),
    )

    assert [configuration["n_chains"] for configuration in configurations] == [1] * 9
    assert result["compile_plus_execution_seconds"] == 1.0
    assert all(
        repetition["raw_wall_seconds"] == [1.0, 1.0, 1.0, 1.0]
        for repetition in result["measured_repetitions"]
    )
    assert all(
        repetition["normalized_wall_seconds_per_chain"] == 1.0
        for repetition in result["measured_repetitions"]
    )
    assert all(
        repetition["total_wall_seconds"] == 4.0
        for repetition in result["measured_repetitions"]
    )


def test_repeated_runner_uses_one_vmapped_gpu_fit_with_requested_chain_count() -> None:
    """Catch GPU vmap mode falling back to sequential single-chain fits."""
    benchmark = _benchmark_module()
    configurations: list[dict[str, object]] = []
    estimators: list[FakeEstimator] = []

    def estimator_factory(**configuration: object) -> FakeEstimator:
        configurations.append(dict(configuration))
        estimator = FakeEstimator(index=len(estimators), fit_keys=[], blocked=[])
        estimators.append(estimator)
        return estimator

    clock_values = iter([0.0, 1.0, 2.0, 10.0])
    result = benchmark.run_repeated_fit_benchmark(
        np.zeros((6, 2), dtype=np.float32),
        np.diag([2.0, 3.0]).astype(np.float32),
        estimator_factory=estimator_factory,
        estimator_kwargs={
            "burnin": 2,
            "n_samples": 2,
            "dtype": "float32",
            "device": "gpu",
        },
        execution_model="vmap",
        chain_count=4,
        repetitions=1,
        seed=41,
        clock=lambda: next(clock_values),
    )

    assert [configuration["n_chains"] for configuration in configurations] == [4, 4]
    assert len(estimators) == 2
    assert result["measured_repetitions"][0]["raw_wall_seconds"] == [8.0]
    assert result["measured_repetitions"][0]["total_wall_seconds"] == 8.0
    assert result["measured_repetitions"][0]["normalized_wall_seconds_per_chain"] == 2.0


@pytest.mark.parametrize(
    ("execution_option", "expected_execution_model"),
    [(None, "sequential"), ("vmap", "vmap")],
)
def test_gpu_cli_forwards_selectable_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
    execution_option: str | None,
    expected_execution_model: str,
) -> None:
    """Catch GPU CLI mode selection being ignored or falling back to sequential."""
    benchmark = _benchmark_module()
    forwarded_execution_models: list[str] = []

    monkeypatch.setattr(
        benchmark,
        "generate_fixture",
        lambda **_: SimpleNamespace(
            observations=np.zeros((6, 2), dtype=np.float64),
            covariance=np.eye(2, dtype=np.float64),
            sha256="0" * 64,
        ),
    )

    def fake_runner(
        observations: np.ndarray,
        truth_covariance: np.ndarray,
        **kwargs: object,
    ) -> dict[str, object]:
        del observations, truth_covariance
        forwarded_execution_models.append(str(kwargs["execution_model"]))
        return {
            "compile_plus_execution_seconds": 1.0,
            "execution_model": expected_execution_model,
            "chain_count": 4,
            "measured_repetitions": [],
            "timing_summary": {
                "median": 0.5,
                "q1": 0.5,
                "q3": 0.5,
                "min": 0.5,
                "max": 0.5,
            },
        }

    monkeypatch.setattr(benchmark, "run_repeated_fit_benchmark", fake_runner)
    monkeypatch.setattr(benchmark, "_git_provenance", lambda _: {})
    monkeypatch.setattr(benchmark, "_environment_metadata", dict)
    arguments = ["--device", "gpu", "--dimensions", "2", "--repetitions", "1"]
    if execution_option is not None:
        arguments.extend(["--execution-mode", execution_option])

    benchmark.main(arguments)

    assert forwarded_execution_models == [expected_execution_model]


def test_cli_selects_bm_and_writes_jsonl_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _benchmark_module()
    selected_factories: list[object] = []
    selected_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr(
        benchmark,
        "generate_fixture",
        lambda **_: SimpleNamespace(
            observations=np.zeros((6, 2), dtype=np.float64),
            covariance=np.eye(2, dtype=np.float64),
            sha256="0" * 64,
        ),
    )

    def fake_runner(
        observations: np.ndarray,
        truth_covariance: np.ndarray,
        **kwargs: object,
    ) -> dict[str, object]:
        del observations, truth_covariance
        selected_factories.append(kwargs["estimator_factory"])
        estimator_kwargs = kwargs["estimator_kwargs"]
        assert isinstance(estimator_kwargs, dict)
        selected_kwargs.append(estimator_kwargs)
        return {
            "compile_plus_execution_seconds": 1.0,
            "execution_model": "parallel",
            "chain_count": 4,
            "measured_repetitions": [],
            "timing_summary": {
                "median": 0.5,
                "q1": 0.5,
                "q3": 0.5,
                "min": 0.5,
                "max": 0.5,
            },
        }

    monkeypatch.setattr(benchmark, "run_repeated_fit_benchmark", fake_runner)
    monkeypatch.setattr(
        benchmark,
        "_git_provenance",
        lambda _: {"revision": "abc123", "dirty": False},
    )
    monkeypatch.setattr(benchmark, "_environment_metadata", dict)
    output_path = tmp_path / "bm.jsonl"

    benchmark.main(
        [
            "--estimator",
            "bm",
            "--dimensions",
            "2",
            "--repetitions",
            "1",
            "--output",
            str(output_path),
        ]
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["benchmark"] == "bm-public-scaling"
    assert records[0]["estimator"] == "bm"
    assert selected_factories == [benchmark.BMSPCov]
    assert selected_kwargs == [
        {
            "burnin": 50,
            "n_samples": 50,
            "dtype": "float32",
            "device": "cpu",
        }
    ]


def test_cli_emits_one_provenance_record_per_requested_dimension(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module()
    fixture_calls: list[tuple[int, float, int, int, str]] = []
    runner_calls: list[tuple[int, int, str, int, int]] = []

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
            observations=np.zeros((n_observations, dimension), dtype=np.float64),
            covariance=np.eye(dimension, dtype=np.float64),
            precision=np.eye(dimension, dtype=np.float64),
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
        runner_calls.append(
            (
                observations.shape[1],
                runner_seed,
                str(kwargs["execution_model"]),
                int(kwargs["chain_count"]),
                int(kwargs["repetitions"]),
            )
        )
        assert observations.dtype == np.float32
        return {
            "compile_plus_execution_seconds": 1.0,
            "execution_model": "parallel",
            "chain_count": 4,
            "measured_repetitions": [],
            "timing_summary": {
                "median": statistics.median([0.5, 0.6]),
                "q1": 0.525,
                "q3": 0.575,
                "min": 0.5,
                "max": 0.6,
            },
        }

    monkeypatch.setattr(benchmark, "generate_fixture", fake_fixture)
    monkeypatch.setattr(benchmark, "run_repeated_fit_benchmark", fake_runner)
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
            "50",
            "--samples",
            "50",
            "--chains",
            "4",
            "--repetitions",
            "10",
            "--seed",
            "19",
        ]
    )

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["dimension"] for record in records] == [25, 50, 100, 200]
    assert fixture_calls == [
        (dimension, 0.05, dimension * 3, 19, "float64")
        for dimension in (25, 50, 100, 200)
    ]
    assert runner_calls == [
        (dimension, 19, "parallel", 4, 10) for dimension in (25, 50, 100, 200)
    ]
    for record in records:
        assert record.keys() >= {
            "benchmark",
            "burnin",
            "chain_count",
            "compile_plus_execution_seconds",
            "density",
            "device",
            "dimension",
            "dtype",
            "environment",
            "execution_model",
            "fixture_sha256",
            "git",
            "measured_repetitions",
            "n_observations",
            "repetitions",
            "prng_policy",
            "samples",
            "schema_version",
            "seed",
            "timing_summary",
        }
        assert record["benchmark"] == "sbm-public-scaling"
        assert record["schema_version"] == "2.0"
        assert record["device"] == "cpu"
        assert record["dtype"] == "float32"
        assert record["density"] == 0.05
        assert record["n_observations"] == record["dimension"] * 3
        assert record["burnin"] == 50
        assert record["samples"] == 50
        assert record["chain_count"] == 4
        assert record["execution_model"] == "parallel"
        assert record["repetitions"] == 10
        assert record["seed"] == 19
        assert record["prng_policy"] == (
            "jax.random.key(seed) is split into one warm-up key and one key per "
            "measured repetition; sequential fits split each repetition key by "
            "chain, while parallel and vmap fits pass one repetition key to the "
            "estimator, which derives per-chain keys."
        )
        assert record["fixture_sha256"] == f"{record['dimension']:064x}"
        assert record["git"] == {"revision": "abc123", "dirty": False}
        assert record["environment"] == {"python": "3.12", "jax": "0.11"}


def test_cli_estimator_fixture_dtype_policy_is_forwarded_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module()
    fixture_dtypes: list[str] = []

    def fake_fixture(**kwargs: object) -> SimpleNamespace:
        fixture_dtypes.append(str(kwargs["dtype"]))
        return SimpleNamespace(
            observations=np.zeros((6, 2), dtype=np.float32),
            covariance=np.eye(2, dtype=np.float32),
            sha256="0" * 64,
        )

    monkeypatch.setattr(benchmark, "generate_fixture", fake_fixture)
    monkeypatch.setattr(
        benchmark,
        "run_repeated_fit_benchmark",
        lambda *_args, **_kwargs: {
            "compile_plus_execution_seconds": 1.0,
            "execution_model": "parallel",
            "chain_count": 1,
            "measured_repetitions": [],
            "timing_summary": {
                "median": 0.5,
                "q1": 0.5,
                "q3": 0.5,
                "min": 0.5,
                "max": 0.5,
            },
        },
    )
    monkeypatch.setattr(benchmark, "_git_provenance", lambda _: {})
    monkeypatch.setattr(benchmark, "_environment_metadata", dict)

    benchmark.main(
        [
            "--dtype",
            "float32",
            "--fixture-dtype-policy",
            "estimator",
            "--dimensions",
            "2",
            "--chains",
            "1",
            "--repetitions",
            "1",
        ]
    )

    [record] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert fixture_dtypes == ["float32"]
    assert record["fixture_dtype_policy"] == "estimator"
