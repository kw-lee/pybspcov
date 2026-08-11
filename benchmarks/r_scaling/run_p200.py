"""Compare p=200 BM chain throughput with R bspcov on one shared fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import numpy.typing as npt

from pybspcov import BMSPCov, __version__ as pybspcov_version

ExecutionModel = Literal["parallel", "sequential"]
EstimatorFactory = Callable[..., Any]
FloatArray = npt.NDArray[np.floating[Any]]


def _generate_fixture(**kwargs: object) -> Any:
    scaling_path = Path(__file__).parents[1] / "sbm_public_scaling.py"
    spec = importlib.util.spec_from_file_location("sbm_public_scaling", scaling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scaling fixture generator from {scaling_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "generate_fixture")(**kwargs)


def normalize_chain_timing(
    wall_seconds: Sequence[float],
    *,
    execution_model: ExecutionModel,
    chain_count: int,
) -> dict[str, object]:
    """Normalize parallel-total or sequential-per-chain wall measurements."""
    if chain_count < 1:
        raise ValueError("chain_count must be positive")
    timings = [float(value) for value in wall_seconds]
    expected_count = 1 if execution_model == "parallel" else chain_count
    if len(timings) != expected_count:
        raise ValueError(
            f"{execution_model} timing requires {expected_count} wall measurement(s)"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in timings):
        raise ValueError("wall measurements must be finite and positive")
    total = float(sum(timings))
    normalized = total / chain_count
    return {
        "execution_model": execution_model,
        "raw_wall_seconds": timings,
        "total_wall_seconds": total,
        "normalized_wall_seconds_per_chain": normalized,
        "chains_per_second": 1.0 / normalized,
    }


def write_shared_fixture(
    output_directory: Path,
    *,
    dimension: int,
    density: float,
    n_observations: int,
    seed: int,
) -> dict[str, object]:
    """Write one exact float64 scaling fixture for both Python and R."""
    fixture = _generate_fixture(
        dimension=dimension,
        density=density,
        n_observations=n_observations,
        seed=seed,
        dtype="float64",
    )
    observations = np.asarray(fixture.observations, dtype=np.float64)
    truth = np.asarray(fixture.covariance, dtype=np.float64)
    initial = np.diag(np.var(observations, axis=0, ddof=1))
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_directory / "observations.csv",
        observations,
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(
        output_directory / "truth_covariance.csv",
        truth,
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(
        output_directory / "initial_covariance.csv",
        initial,
        delimiter=",",
        fmt="%.17g",
    )
    return {
        "dimension": dimension,
        "n_observations": n_observations,
        "density": density,
        "seed": seed,
        "dtype": "float64",
        "fixture_sha256": fixture.sha256,
    }


def _posterior_summary(draws: FloatArray, truth: FloatArray) -> dict[str, object]:
    dimension = truth.shape[0]
    flattened = np.asarray(draws).reshape(-1, dimension, dimension)
    posterior_mean = np.mean(flattened, axis=0)
    finite = bool(np.all(np.isfinite(posterior_mean)))
    symmetric = bool(np.allclose(posterior_mean, posterior_mean.T))
    spd = bool(
        finite
        and symmetric
        and np.all(np.linalg.eigvalsh(posterior_mean) > 0.0)
    )
    truth_norm = float(np.linalg.norm(truth))
    error = float(np.linalg.norm(posterior_mean - truth) / truth_norm)
    return {
        "retained_draws": int(flattened.shape[0]),
        "posterior_mean_finite": finite,
        "posterior_mean_symmetric": symmetric,
        "posterior_mean_spd": spd,
        "truth_relative_frobenius_error": error,
    }


def measure_python_mode(
    observations: FloatArray,
    truth: FloatArray,
    initial_covariance: FloatArray,
    *,
    device: Literal["cpu", "gpu"],
    dtype: Literal["float32", "float64"],
    burnin: int,
    n_samples: int,
    chain_count: int,
    seed: int,
    estimator_factory: EstimatorFactory = BMSPCov,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Measure one vmapped CPU fit or sequential single-chain GPU fits."""
    estimator_chains = chain_count if device == "cpu" else 1

    def fit_once(key_seed: int) -> FloatArray:
        fitted = estimator_factory(
            n_samples=n_samples,
            burnin=burnin,
            n_chains=estimator_chains,
            dtype=dtype,
            device=device,
        ).fit(
            observations,
            key=jax.random.key(key_seed),
            initial_covariance=initial_covariance,
        )
        return np.asarray(fitted.posterior_samples_)

    compile_start = clock()
    fit_once(seed)
    compile_seconds = clock() - compile_start

    if device == "cpu":
        measured_start = clock()
        draws = fit_once(seed + 1)
        measured_seconds = clock() - measured_start
        timing = normalize_chain_timing(
            [measured_seconds],
            execution_model="parallel",
            chain_count=chain_count,
        )
    else:
        chain_draws = []
        wall_seconds = []
        for chain_index in range(chain_count):
            measured_start = clock()
            chain_draws.append(fit_once(seed + chain_index + 1))
            wall_seconds.append(clock() - measured_start)
        draws = np.concatenate(chain_draws, axis=0)
        timing = normalize_chain_timing(
            wall_seconds,
            execution_model="sequential",
            chain_count=chain_count,
        )

    return {
        "implementation": "pybspcov",
        "device": device,
        "dtype": dtype,
        "burnin": burnin,
        "n_samples": n_samples,
        "chain_count": chain_count,
        "compile_plus_execution_seconds": float(compile_seconds),
        **timing,
        **_posterior_summary(draws, truth),
    }


def _git_provenance(project_root: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ("git", "-C", str(project_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "-C", str(project_root), "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "<unavailable>", "dirty": "<unavailable>"}


def _run_r(
    fixture_directory: Path,
    *,
    r_library: Path,
    fixture_sha256: str,
    burnin: int,
    n_samples: int,
    chain_count: int,
    seed: int,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["R_LIBS"] = str(r_library)
    script = Path(__file__).with_name("run_bspcov.R")
    result = subprocess.run(
        (
            "Rscript",
            str(script),
            "--fixture-dir",
            str(fixture_directory),
            "--fixture-sha256",
            fixture_sha256,
            "--burnin",
            str(burnin),
            "--n-samples",
            str(n_samples),
            "--n-chains",
            str(chain_count),
            "--seed",
            str(seed),
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("R runner produced no JSON output")
    parsed = json.loads(lines[-1])
    if not isinstance(parsed, dict):
        raise RuntimeError("R runner output must be a JSON object")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=200)
    parser.add_argument("--n-factor", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--burnin", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    arguments = parser.parse_args(argv)
    if arguments.dimension < 2 or arguments.n_factor < 1:
        parser.error("dimension must be at least two and n-factor must be positive")
    if arguments.burnin < 0 or arguments.n_samples < 1 or arguments.n_chains < 1:
        parser.error("burnin must be non-negative; samples and chains must be positive")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="pybspcov-r-scaling-") as temporary:
        fixture_directory = Path(temporary)
        fixture = write_shared_fixture(
            fixture_directory,
            dimension=arguments.dimension,
            density=arguments.density,
            n_observations=arguments.dimension * arguments.n_factor,
            seed=arguments.seed,
        )
        observations = np.loadtxt(
            fixture_directory / "observations.csv", delimiter=","
        )
        truth = np.loadtxt(
            fixture_directory / "truth_covariance.csv", delimiter=","
        )
        initial = np.loadtxt(
            fixture_directory / "initial_covariance.csv", delimiter=","
        )
        r_result = _run_r(
            fixture_directory,
            r_library=arguments.r_library,
            fixture_sha256=str(fixture["fixture_sha256"]),
            burnin=arguments.burnin,
            n_samples=arguments.n_samples,
            chain_count=arguments.n_chains,
            seed=arguments.seed,
        )
        python_results = [
            measure_python_mode(
                observations,
                truth,
                initial,
                device=device,
                dtype=dtype,
                burnin=arguments.burnin,
                n_samples=arguments.n_samples,
                chain_count=arguments.n_chains,
                seed=arguments.seed,
            )
            for device, dtype in (
                ("cpu", "float64"),
                ("gpu", "float64"),
                ("cpu", "float32"),
                ("gpu", "float32"),
            )
        ]

    output = {
        "benchmark": "p200-r-bm-comparison",
        "schema_version": "1.0",
        "fixture": fixture,
        "configuration": {
            "burnin": arguments.burnin,
            "n_samples": arguments.n_samples,
            "chain_count": arguments.n_chains,
            "seed": arguments.seed,
        },
        "git": _git_provenance(project_root),
        "environment": {
            "python": platform.python_version(),
            "pybspcov": pybspcov_version,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "r": r_result,
        "python": python_results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
