"""Measure end-to-end scaling through the public :class:`SBMSPCov` API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, TextIO

import jax
import jaxlib
import numpy as np
import numpy.typing as npt

from pybspcov import BMSPCov, SBMSPCov

FloatArray = npt.NDArray[np.floating[Any]]


class ScalingFixture(NamedTuple):
    """Deterministic host-side sparse Gaussian scaling fixture."""

    precision: FloatArray
    covariance: FloatArray
    observations: FloatArray
    sha256: str


def _fixture_sha256(
    precision: FloatArray,
    covariance: FloatArray,
    observations: FloatArray,
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


def generate_fixture(
    *,
    dimension: int,
    density: float,
    n_observations: int,
    seed: int,
    dtype: str,
) -> ScalingFixture:
    """Generate a centered Gaussian fixture with sparse SPD precision."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between zero and one")
    if n_observations < 2:
        raise ValueError("n_observations must be at least two")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    value_dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    lower_rows, lower_columns = np.tril_indices(dimension, k=-1)
    possible_edges = lower_rows.size
    edge_count = round(density * possible_edges)
    selected = rng.choice(possible_edges, size=edge_count, replace=False)
    weights = rng.uniform(0.05, 0.2, size=edge_count)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=edge_count)

    precision = np.zeros((dimension, dimension), dtype=np.float64)
    precision[lower_rows[selected], lower_columns[selected]] = weights * signs
    precision[lower_columns[selected], lower_rows[selected]] = weights * signs
    diagonal = np.sum(np.abs(precision), axis=1) + 1.0
    np.fill_diagonal(precision, diagonal)
    precision = np.ascontiguousarray(precision, dtype=value_dtype)
    covariance = np.ascontiguousarray(np.linalg.inv(precision), dtype=value_dtype)

    standard = rng.normal(size=(n_observations, dimension)).astype(
        value_dtype,
        copy=False,
    )
    observations = standard @ np.linalg.cholesky(covariance).T
    observations -= observations.mean(axis=0, keepdims=True)
    observations = np.ascontiguousarray(observations, dtype=value_dtype)
    return ScalingFixture(
        precision=precision,
        covariance=covariance,
        observations=observations,
        sha256=_fixture_sha256(precision, covariance, observations),
    )


def _block_public_outputs(estimator: Any) -> None:
    outputs = [
        estimator.covariance_,
        estimator.posterior_samples_packed_,
        estimator.phi_samples_packed_,
    ]
    screening_mask = getattr(estimator, "screening_mask_", None)
    if screening_mask is not None:
        outputs.append(screening_mask)
    outputs.append(estimator.diagnostics_.accepted)
    for output in outputs:
        for leaf in jax.tree.leaves(output):
            blocker = getattr(leaf, "block_until_ready", None)
            if blocker is not None:
                blocker()


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("timings must be finite positive values")
    quantiles = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    return {
        "median": float(statistics.median(values)),
        "q1": float(quantiles[0]),
        "q3": float(quantiles[1]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _measured_repetition(
    estimators: Sequence[Any],
    raw_wall_seconds: Sequence[float],
    truth: npt.NDArray[Any],
    *,
    execution_model: Literal["parallel", "sequential", "vmap"],
    chain_count: int,
    n_samples: int,
    repetition_index: int,
) -> dict[str, object]:
    total_wall_seconds = float(sum(raw_wall_seconds))
    normalized_wall_seconds = total_wall_seconds / chain_count
    covariance = np.mean(
        np.stack([np.asarray(estimator.covariance_) for estimator in estimators]),
        axis=0,
    )
    finite = bool(np.all(np.isfinite(covariance)))
    symmetric = bool(np.allclose(covariance, covariance.T))
    spd = bool(finite and symmetric and np.all(np.linalg.eigvalsh(covariance) > 0.0))
    if not (finite and symmetric and spd):
        raise RuntimeError("public posterior mean must be finite, symmetric, and SPD")

    truth_norm = float(np.linalg.norm(truth))
    accepted = sum(
        int(np.count_nonzero(np.asarray(estimator.diagnostics_.accepted)))
        for estimator in estimators
    )
    rejected = sum(
        int(estimator.diagnostics_.n_rejected_sweeps) for estimator in estimators
    )
    return {
        "repetition": repetition_index,
        "execution_model": execution_model,
        "raw_wall_seconds": [float(value) for value in raw_wall_seconds],
        "total_wall_seconds": total_wall_seconds,
        "normalized_wall_seconds_per_chain": normalized_wall_seconds,
        "chains_per_second": 1.0 / normalized_wall_seconds,
        "retained_draws": n_samples * chain_count,
        "posterior_mean_finite": finite,
        "posterior_mean_symmetric": symmetric,
        "posterior_mean_spd": spd,
        "truth_relative_frobenius_error": float(
            np.linalg.norm(covariance - truth) / truth_norm
        ),
        "accepted_sweeps": accepted,
        "rejected_sweeps": rejected,
    }


def run_repeated_fit_benchmark(
    observations: npt.NDArray[Any],
    truth_covariance: npt.NDArray[Any],
    *,
    estimator_factory: Callable[..., Any] = SBMSPCov,
    estimator_kwargs: Mapping[str, object],
    execution_model: Literal["parallel", "sequential", "vmap"],
    chain_count: int,
    repetitions: int,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Compile once, then record repetitions at the requested chain count."""
    if chain_count < 1 or repetitions < 1:
        raise ValueError("chain_count and repetitions must be positive")
    if execution_model not in {"parallel", "sequential", "vmap"}:
        raise ValueError("execution_model must be parallel, sequential, or vmap")
    if "n_samples" not in estimator_kwargs:
        raise ValueError("estimator_kwargs must include n_samples")

    x = np.asarray(observations)
    truth = np.asarray(truth_covariance)
    if x.ndim != 2:
        raise ValueError("observations must be a two-dimensional array")
    dimension = x.shape[1]
    if truth.shape != (dimension, dimension):
        raise ValueError(f"truth_covariance must have shape ({dimension}, {dimension})")
    truth_norm = float(np.linalg.norm(truth))
    if not np.isfinite(truth_norm) or truth_norm <= 0.0:
        raise ValueError("truth_covariance must have a finite positive norm")

    uses_vmapped_fit = execution_model in {"parallel", "vmap"}
    estimator_chain_count = chain_count if uses_vmapped_fit else 1
    configuration = dict(estimator_kwargs)
    configuration["n_chains"] = estimator_chain_count
    repetition_keys = jax.random.split(jax.random.key(seed), repetitions + 1)

    compile_estimator = estimator_factory(**configuration)
    compile_start = clock()
    compiled = compile_estimator.fit(x, key=repetition_keys[0])
    _block_public_outputs(compiled)
    compile_seconds = clock() - compile_start
    if not np.isfinite(compile_seconds) or compile_seconds <= 0.0:
        raise RuntimeError("compile timing must be finite and positive")

    measured_repetitions: list[dict[str, object]] = []
    for repetition_index, repetition_key in enumerate(repetition_keys[1:]):
        fit_keys = (
            [repetition_key]
            if uses_vmapped_fit
            else list(jax.random.split(repetition_key, chain_count))
        )
        estimators: list[Any] = []
        raw_wall_seconds: list[float] = []
        for fit_key in fit_keys:
            estimator = estimator_factory(**configuration)
            start = clock()
            fitted = estimator.fit(x, key=fit_key)
            _block_public_outputs(fitted)
            elapsed = clock() - start
            if not np.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError("fit timings must be finite positive values")
            estimators.append(fitted)
            raw_wall_seconds.append(float(elapsed))
        measured_repetitions.append(
            _measured_repetition(
                estimators,
                raw_wall_seconds,
                truth,
                execution_model=execution_model,
                chain_count=chain_count,
                n_samples=int(estimator_kwargs["n_samples"]),
                repetition_index=repetition_index,
            )
        )

    normalized_timings = [
        float(repetition["normalized_wall_seconds_per_chain"])
        for repetition in measured_repetitions
    ]
    return {
        "compile_plus_execution_seconds": float(compile_seconds),
        "execution_model": execution_model,
        "chain_count": chain_count,
        "measured_repetitions": measured_repetitions,
        "timing_summary": _distribution_summary(normalized_timings),
    }


def _git_provenance(project_root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(project_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()

    try:
        return {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"revision": "<unavailable>", "dirty": "<unavailable>"}


def _environment_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": str(getattr(jaxlib, "__version__", "unknown")),
        "numpy": np.__version__,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("bm", "sbm"), default="sbm")
    parser.add_argument("--device", choices=("cpu", "gpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--fixture-dtype-policy",
        choices=("float64", "estimator"),
        default="float64",
        help="fixture generation dtype; defaults to the current float64 policy",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200],
    )
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--n-factor", type=int, default=3)
    parser.add_argument("--burnin", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--execution-mode",
        choices=("sequential", "vmap"),
        help="GPU execution mode; defaults to sequential for historical compatibility",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if any(dimension < 2 for dimension in arguments.dimensions):
        parser.error("all dimensions must be at least two")
    if not 0.0 <= arguments.density <= 1.0:
        parser.error("--density must be between zero and one")
    if arguments.n_factor < 1:
        parser.error("--n-factor must be positive")
    if arguments.burnin < 0:
        parser.error("--burnin must be non-negative")
    if arguments.samples < 1 or arguments.chains < 1 or arguments.repetitions < 1:
        parser.error("--samples, --chains, and --repetitions must be positive")
    if arguments.device == "cpu" and arguments.execution_mode is not None:
        parser.error("--execution-mode is available only for GPU or CUDA")
    if arguments.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 requires JAX_ENABLE_X64=1")
    return arguments


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> None:
    """Run requested dimensions and emit one JSON record per public fit cell."""
    arguments = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    git = _git_provenance(project_root)
    environment = _environment_metadata()
    estimator_factory = BMSPCov if arguments.estimator == "bm" else SBMSPCov
    execution_model: Literal["parallel", "sequential", "vmap"] = (
        "parallel"
        if arguments.device == "cpu"
        else "sequential"
        if arguments.execution_mode is None
        else arguments.execution_mode
    )
    estimator_kwargs: dict[str, object] = {
        "burnin": arguments.burnin,
        "n_samples": arguments.samples,
        "dtype": arguments.dtype,
        "device": arguments.device,
    }
    if arguments.estimator == "sbm":
        estimator_kwargs.update(
            cutoff_method="correlation",
            retained_fraction=arguments.density,
        )

    output_path = arguments.output
    if output_path is not None and stdout is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle: TextIO = output_path.open("w", encoding="utf-8")
        close_output = True
    else:
        output_handle = sys.stdout if stdout is None else stdout
        close_output = False

    try:
        for dimension in arguments.dimensions:
            n_observations = dimension * arguments.n_factor
            fixture_dtype = (
                arguments.dtype
                if arguments.fixture_dtype_policy == "estimator"
                else "float64"
            )
            fixture = generate_fixture(
                dimension=dimension,
                density=arguments.density,
                n_observations=n_observations,
                seed=arguments.seed,
                dtype=fixture_dtype,
            )
            value_dtype = np.dtype(arguments.dtype)
            summary = run_repeated_fit_benchmark(
                np.asarray(fixture.observations, dtype=value_dtype),
                np.asarray(fixture.covariance, dtype=value_dtype),
                estimator_factory=estimator_factory,
                estimator_kwargs=estimator_kwargs,
                execution_model=execution_model,
                chain_count=arguments.chains,
                repetitions=arguments.repetitions,
                seed=arguments.seed,
            )
            record = {
                "benchmark": f"{arguments.estimator}-public-scaling",
                "schema_version": "2.0",
                "estimator": arguments.estimator,
                "dimension": dimension,
                "density": arguments.density,
                "n_observations": n_observations,
                "burnin": arguments.burnin,
                "samples": arguments.samples,
                "chain_count": arguments.chains,
                "repetitions": arguments.repetitions,
                "seed": arguments.seed,
                "prng_policy": (
                    "jax.random.key(seed) is split into one warm-up key and one key "
                    "per measured repetition; sequential fits split each repetition "
                    "key by chain, while parallel and vmap fits pass one repetition "
                    "key to the estimator, which derives per-chain keys."
                ),
                "device": arguments.device,
                "dtype": arguments.dtype,
                "fixture_dtype_policy": arguments.fixture_dtype_policy,
                "fixture_sha256": fixture.sha256,
                "git": git,
                "environment": environment,
                **summary,
            }
            print(json.dumps(record, sort_keys=True), file=output_handle)
            output_handle.flush()
    finally:
        if close_output:
            output_handle.close()


if __name__ == "__main__":
    main()
