#!/usr/bin/env python3
"""Run the pybspcov BM sampler on the committed upstream R example."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pybspcov import __version__ as pybspcov_version
from pybspcov.kernels.bm import initialize_bm_state, sample_bm_chain

FIXTURE_SEED = 1
SAMPLER_SEED = 1
QUANTILE_PROBABILITIES = (0.025, 0.5, 0.975)
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def summarize_draws(draws: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Summarize covariance draws and estimate MCSE with contiguous batches."""
    draws = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1] != draws.shape[2]:
        raise ValueError("draws must have shape (n_samples, p, p)")
    if truth.shape != draws.shape[1:]:
        raise ValueError("truth must match the covariance draw shape")
    n_samples = draws.shape[0]
    if n_samples < 4:
        raise ValueError("at least 4 covariance draws are required")

    n_batches = min(20, n_samples // 2)
    batch_size = n_samples // n_batches
    trimmed_samples = n_batches * batch_size
    batches = draws[:trimmed_samples].reshape(
        n_batches, batch_size, draws.shape[1], draws.shape[2]
    )

    posterior_mean = np.mean(draws, axis=0)
    posterior_sd = np.std(draws, axis=0, ddof=1)
    quantiles = np.quantile(draws, QUANTILE_PROBABILITIES, axis=0, method="linear")
    batch_means = np.mean(batches, axis=1)
    batch_sds = np.std(batches, axis=1, ddof=1)
    batch_quantiles = np.stack(
        [
            np.quantile(batch, QUANTILE_PROBABILITIES, axis=0, method="linear")
            for batch in batches
        ]
    )
    batch_rmses = np.sqrt(np.mean(np.square(batch_means - truth), axis=(1, 2)))

    def batch_mcse(values: np.ndarray) -> np.ndarray:
        return np.std(values, axis=0, ddof=1) / np.sqrt(n_batches)

    return {
        "posterior_mean": posterior_mean,
        "posterior_mean_mcse": batch_mcse(batch_means),
        "posterior_sd": posterior_sd,
        "posterior_sd_mcse": batch_mcse(batch_sds),
        "q025": quantiles[0],
        "q025_mcse": batch_mcse(batch_quantiles[:, 0]),
        "q50": quantiles[1],
        "q50_mcse": batch_mcse(batch_quantiles[:, 1]),
        "q975": quantiles[2],
        "q975_mcse": batch_mcse(batch_quantiles[:, 2]),
        "rmse": float(np.sqrt(np.mean(np.square(posterior_mean - truth)))),
        "rmse_mcse": float(batch_mcse(batch_rmses)),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "trimmed_samples": trimmed_samples,
    }


def validate_chain_output(accepted: Any, draws: Any) -> None:
    """Reject incomplete sampler runs before publishing benchmark output."""
    accepted_host = np.asarray(jax.device_get(accepted), dtype=np.bool_)
    if not np.all(accepted_host):
        rejected = int(accepted_host.size - np.count_nonzero(accepted_host))
        raise RuntimeError(f"BM chain contained {rejected} rejected sweep(s)")
    draws_host = np.asarray(jax.device_get(draws))
    if not np.all(np.isfinite(draws_host)):
        raise RuntimeError("BM chain produced nonfinite covariance draws")


def detect_cpu_model() -> str:
    """Return a useful CPU identity on Linux and portable fallbacks elsewhere."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition(":")
                if (
                    separator
                    and name.strip() in {"model name", "Hardware", "Processor"}
                    and value.strip()
                ):
                    return value.strip()
        except OSError:
            pass
    return platform.processor() or f"unavailable (architecture: {platform.machine()})"


def detect_physical_cores() -> str:
    """Return Linux physical-core count when topology data are available."""
    if sys.platform.startswith("linux"):
        try:
            records = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")
        except OSError:
            records = []
        core_pairs: set[tuple[str, str]] = set()
        for record in records:
            fields = {}
            for line in record.splitlines():
                name, separator, value = line.partition(":")
                if separator:
                    fields[name.strip()] = value.strip()
            if "physical id" in fields and "core id" in fields:
                core_pairs.add((fields["physical id"], fields["core id"]))
        if core_pairs:
            return str(len(core_pairs))
    return "unavailable"


def select_device(platform_name: str) -> jax.Device:
    """Select one requested JAX device or fail with an actionable message."""
    try:
        devices = jax.devices(platform_name)
    except RuntimeError as error:
        raise RuntimeError(
            f"requested JAX {platform_name.upper()} device is unavailable"
        ) from error
    if not devices:
        raise RuntimeError(
            f"requested JAX {platform_name.upper()} device is unavailable"
        )
    return devices[0]


def _parse_positive_integer(value: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if str(parsed) != value or parsed < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}")
    return parsed


def _parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burnin",
        type=lambda value: _parse_positive_integer(value, minimum=0),
        default=1000,
    )
    parser.add_argument(
        "--n-samples",
        type=lambda value: _parse_positive_integer(value, minimum=4),
        default=1000,
    )
    parser.add_argument(
        "--repetitions",
        type=lambda value: _parse_positive_integer(value, minimum=1),
        default=5,
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=script_directory / "results")
    return parser.parse_args()


def _read_matrix(directory: Path, filename: str) -> np.ndarray:
    return np.loadtxt(directory / filename, delimiter=",", dtype=np.float64)


def _other_indices(dimension: int) -> np.ndarray:
    return np.asarray(
        [
            [index for index in range(dimension) if index != column]
            for column in range(dimension)
        ],
        dtype=np.int32,
    )


def _metadata(
    *,
    device: jax.Device,
    n: int,
    p: int,
    burnin: int,
    n_samples: int,
    repetitions: int,
    summary: dict[str, Any],
) -> list[tuple[str, str]]:
    thread_environment = [
        (name, os.environ.get(name, "<unset>")) for name in THREAD_ENVIRONMENT_VARIABLES
    ]
    metadata = [
        ("implementation", "pybspcov"),
        ("package", "pybspcov"),
        ("package_version", pybspcov_version),
        ("jax_version", jax.__version__),
        ("jaxlib_version", jax.lib.__version__),
        ("python_version", platform.python_version()),
        ("platform", platform.platform()),
        ("dtype", "float64"),
        ("device", device.platform.upper()),
        ("device_id", str(device.id)),
        ("device_kind", device.device_kind),
        ("jax_backend", device.platform),
        ("cpu_model", detect_cpu_model()),
        ("logical_cores", str(os.cpu_count() or "unavailable")),
        ("physical_cores", detect_physical_cores()),
        *thread_environment,
        ("n", str(n)),
        ("p", str(p)),
        ("burnin", str(burnin)),
        ("n_samples", str(n_samples)),
        ("chains", "1"),
        ("repetitions", str(repetitions)),
        ("fixture_seed", str(FIXTURE_SEED)),
        ("sampler_seed", str(SAMPLER_SEED)),
        ("fixture_centered", "true"),
        ("n_batches", str(summary["n_batches"])),
        ("batch_size", str(summary["batch_size"])),
        ("trimmed_samples", str(summary["trimmed_samples"])),
        (
            "compile_plus_execution_timing_scope",
            "first jitted BM chain call including compilation, execution, and synchronization",
        ),
        (
            "steady_state_timing_scope",
            "median warmed jitted BM chain execution with a fresh PRNG key and synchronization",
        ),
        (
            "end_to_end_timing_scope",
            "sum of pre-warm fixture, first-chain, and summary work plus post-warm summary and metadata CSV writes; excludes warmed repetitions and the final timing CSV write",
        ),
    ]
    return metadata


def _write_summary(path: Path, summary: dict[str, Any], truth: np.ndarray) -> None:
    fields = (
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
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for column in range(truth.shape[1]):
            for row in range(truth.shape[0]):
                writer.writerow(
                    {
                        "implementation": "pybspcov",
                        "row": row + 1,
                        "column": column + 1,
                        **{
                            field: summary[field][row, column] for field in fields[3:13]
                        },
                        "truth": truth[row, column],
                        "rmse": summary["rmse"],
                        "rmse_mcse": summary["rmse_mcse"],
                    }
                )


def main() -> None:
    arguments = _parse_args()
    jax.config.update("jax_platforms", arguments.device)
    jax.config.update("jax_enable_x64", True)
    device = select_device(arguments.device)
    end_to_end_start = time.perf_counter()
    fixture_directory = Path(__file__).resolve().parent / "data"
    x_numpy = _read_matrix(fixture_directory, "bm_example_x.csv")
    truth = _read_matrix(fixture_directory, "bm_example_truth.csv")
    initial_covariance_numpy = _read_matrix(fixture_directory, "bm_example_initial.csv")
    n, p = x_numpy.shape
    tau1sq_value = 10_000.0 / (n * p**4)

    with jax.default_device(device):
        x = jnp.asarray(x_numpy, dtype=jnp.float64)
        covariance = jnp.asarray(initial_covariance_numpy, dtype=jnp.float64)
        scatter = x.T @ x
        tau1sq = jnp.asarray(tau1sq_value, dtype=jnp.float64)
        initial_state = initialize_bm_state(covariance, tau1sq)
        indices = jnp.asarray(_other_indices(p), dtype=jnp.int32)
        run_chain = jax.jit(
            sample_bm_chain,
            static_argnames=("burnin", "n_samples"),
        )
        sampler_arguments = (
            initial_state,
            scatter,
            indices,
            jnp.asarray(n),
            jnp.asarray(0.5, dtype=jnp.float64),
            jnp.asarray(0.5, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            tau1sq,
        )

        compile_start = time.perf_counter()
        first_result = run_chain(
            jax.random.key(SAMPLER_SEED),
            *sampler_arguments,
            burnin=arguments.burnin,
            n_samples=arguments.n_samples,
        )
        jax.block_until_ready(first_result)
        compile_plus_execution_seconds = time.perf_counter() - compile_start

        validate_chain_output(first_result.accepted, first_result.covariance)

    draws = np.asarray(jax.device_get(first_result.covariance))
    summary = summarize_draws(draws, truth)
    metadata_rows = _metadata(
        device=device,
        n=n,
        p=p,
        burnin=arguments.burnin,
        n_samples=arguments.n_samples,
        repetitions=arguments.repetitions,
        summary=summary,
    )
    pre_warm_seconds = time.perf_counter() - end_to_end_start

    with jax.default_device(device):
        warmed_seconds = []
        for repetition in range(arguments.repetitions):
            warmed_start = time.perf_counter()
            warmed_result = run_chain(
                jax.random.key(SAMPLER_SEED + repetition + 1),
                *sampler_arguments,
                burnin=arguments.burnin,
                n_samples=arguments.n_samples,
            )
            jax.block_until_ready(warmed_result)
            elapsed = time.perf_counter() - warmed_start
            validate_chain_output(warmed_result.accepted, warmed_result.covariance)
            warmed_seconds.append(elapsed)

    output_write_start = time.perf_counter()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(arguments.output_dir / "pybspcov_summary.csv", summary, truth)
    with (arguments.output_dir / "pybspcov_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("name", "value"))
        writer.writerows(metadata_rows)
    output_write_seconds = time.perf_counter() - output_write_start
    end_to_end_seconds = pre_warm_seconds + output_write_seconds

    with (arguments.output_dir / "pybspcov_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "implementation",
                "compile_plus_execution_seconds",
                "steady_state_seconds",
                "steady_state_min_seconds",
                "steady_state_max_seconds",
                "end_to_end_seconds",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "implementation": "pybspcov",
                "compile_plus_execution_seconds": compile_plus_execution_seconds,
                "steady_state_seconds": statistics.median(warmed_seconds),
                "steady_state_min_seconds": min(warmed_seconds),
                "steady_state_max_seconds": max(warmed_seconds),
                "end_to_end_seconds": end_to_end_seconds,
            }
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"run_pybspcov.py: {error}", file=sys.stderr)
        raise SystemExit(2) from error
