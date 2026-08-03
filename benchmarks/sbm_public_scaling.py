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
from typing import Any, NamedTuple, TextIO

import jax
import jaxlib
import numpy as np
import numpy.typing as npt

from pybspcov import SBMSPCov

FloatArray = npt.NDArray[np.floating[Any]]
MemoryProbe = Callable[[str], int]


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
    outputs = (
        estimator.covariance_,
        estimator.posterior_samples_packed_,
        estimator.phi_samples_packed_,
        estimator.screening_mask_,
        estimator.diagnostics_.accepted,
    )
    for output in outputs:
        for leaf in jax.tree.leaves(output):
            blocker = getattr(leaf, "block_until_ready", None)
            if blocker is not None:
                blocker()


def _timing_summary(seconds: Sequence[float]) -> dict[str, object]:
    if not seconds or any(not np.isfinite(value) or value <= 0.0 for value in seconds):
        raise RuntimeError("warmed fit timings must be finite positive values")
    return {
        "raw": list(seconds),
        "median": statistics.median(seconds),
        "min": min(seconds),
        "max": max(seconds),
    }


def run_public_fit_benchmark(
    observations: npt.NDArray[Any],
    truth_covariance: npt.NDArray[Any],
    *,
    estimator_factory: Callable[..., Any] = SBMSPCov,
    estimator_kwargs: Mapping[str, object],
    repetitions: int,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
    memory_probe: MemoryProbe | None = None,
) -> dict[str, object]:
    """Time fresh public estimator fits and validate the final public outputs."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
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

    memory_before = memory_probe("before") if memory_probe is not None else None
    fit_keys = jax.random.split(jax.random.key(seed), repetitions + 1)
    estimators: list[Any] = []
    first_fit_seconds: float | None = None
    timings: list[float] = []
    for fit_index, key in enumerate(fit_keys):
        estimator = estimator_factory(**dict(estimator_kwargs))
        start = clock()
        fitted = estimator.fit(x, key=key)
        _block_public_outputs(fitted)
        elapsed = clock() - start
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError("fit timings must be finite positive values")
        estimators.append(fitted)
        if fit_index == 0:
            first_fit_seconds = elapsed
        else:
            timings.append(elapsed)

    memory_after = memory_probe("after") if memory_probe is not None else None
    memory_peak = memory_probe("peak") if memory_probe is not None else None
    final = estimators[-1]
    covariance = np.asarray(final.covariance_)
    finite = bool(np.all(np.isfinite(covariance)))
    symmetric = bool(np.allclose(covariance, covariance.T))
    spd = bool(finite and symmetric and np.all(np.linalg.eigvalsh(covariance) > 0.0))
    if not (finite and symmetric and spd):
        raise RuntimeError(
            "public SBM posterior mean must be finite, symmetric, and SPD"
        )

    accepted = np.asarray(final.diagnostics_.accepted, dtype=np.bool_)
    screening_mask = np.asarray(final.screening_mask_, dtype=np.bool_)
    compact_width = int(np.max(np.count_nonzero(screening_mask, axis=0), initial=0))
    dtype_name = str(np.dtype(final.dtype_))
    relative_error = float(np.linalg.norm(covariance - truth) / truth_norm)
    if first_fit_seconds is None:
        raise RuntimeError("the first fit timing was not recorded")
    return {
        "first_fit_seconds": first_fit_seconds,
        "warmed_fit_seconds": _timing_summary(timings),
        "posterior_mean_finite": finite,
        "posterior_mean_symmetric": symmetric,
        "posterior_mean_spd": spd,
        "dtype": dtype_name,
        "accepted_sweeps": int(np.count_nonzero(accepted)),
        "rejected_sweeps": int(final.diagnostics_.n_rejected_sweeps),
        "active_edges": int(final.diagnostics_.n_active_edges),
        "compact_width": compact_width,
        "truth_relative_frobenius_error": relative_error,
        "device_memory_bytes": (
            {"before": memory_before, "after": memory_after, "peak": memory_peak}
            if memory_probe is not None
            else None
        ),
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
    parser.add_argument("--device", choices=("cpu", "gpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200],
    )
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--n-factor", type=int, default=3)
    parser.add_argument("--burnin", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    arguments = parser.parse_args(argv)
    if any(dimension < 2 for dimension in arguments.dimensions):
        parser.error("all dimensions must be at least two")
    if not 0.0 <= arguments.density <= 1.0:
        parser.error("--density must be between zero and one")
    if arguments.n_factor < 1:
        parser.error("--n-factor must be positive")
    if arguments.burnin < 0:
        parser.error("--burnin must be non-negative")
    if arguments.samples < 1 or arguments.repetitions < 1:
        parser.error("--samples and --repetitions must be positive")
    if arguments.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 requires JAX_ENABLE_X64=1")
    return arguments


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> None:
    """Run requested dimensions and emit one JSON record per public fit cell."""
    arguments = parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    project_root = Path(__file__).resolve().parents[1]
    git = _git_provenance(project_root)
    environment = _environment_metadata()
    for dimension in arguments.dimensions:
        n_observations = dimension * arguments.n_factor
        fixture = generate_fixture(
            dimension=dimension,
            density=arguments.density,
            n_observations=n_observations,
            seed=arguments.seed,
            dtype=arguments.dtype,
        )
        summary = run_public_fit_benchmark(
            fixture.observations,
            fixture.covariance,
            estimator_factory=SBMSPCov,
            estimator_kwargs={
                "burnin": arguments.burnin,
                "n_samples": arguments.samples,
                "n_chains": 1,
                "cutoff_method": "correlation",
                "retained_fraction": arguments.density,
                "dtype": arguments.dtype,
                "device": arguments.device,
            },
            repetitions=arguments.repetitions,
            seed=arguments.seed,
            memory_probe=None,
        )
        record = {
            "benchmark": "sbm-public-scaling",
            "schema_version": "1.0",
            "dimension": dimension,
            "density": arguments.density,
            "n_observations": n_observations,
            "burnin": arguments.burnin,
            "samples": arguments.samples,
            "repetitions": arguments.repetitions,
            "seed": arguments.seed,
            "device": arguments.device,
            "dtype": arguments.dtype,
            "fixture_sha256": fixture.sha256,
            "git": git,
            "environment": environment,
            **summary,
        }
        print(json.dumps(record, sort_keys=True), file=output)


if __name__ == "__main__":
    main()
