#!/usr/bin/env python3
"""Run one pre-registered pybspcov benchmark cell and emit JSONL."""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
from core import load_manifest, needs_additional_repetitions, validate_timing_record
from fixtures import fixture_sha256, load_fixture

from pybspcov import BandPPP, BMSPCov, SBMSPCov, ThresholdPPP, __version__


def build_estimator(
    method: str,
    *,
    dimension: int,
    dtype: str,
    device: str,
    parallelism: int,
    manifest: Mapping[str, Any],
) -> Any:
    """Construct one estimator from the immutable protocol manifest."""
    methods = manifest["methods"]
    configuration = methods[method]
    common = {"dtype": dtype, "device": device, "n_chains": parallelism}
    if method == "bm":
        return BMSPCov(
            burnin=configuration["burnin"],
            n_samples=configuration["samples"],
            **common,
        )
    if method == "sbm":
        return SBMSPCov(
            burnin=configuration["burnin"],
            n_samples=configuration["samples"],
            cutoff_method=configuration["cutoff_method"],
            retained_fraction=configuration["retained_fraction"],
            **common,
        )
    if method == "bandppp":
        return BandPPP(
            bandwidth=max(1, dimension // configuration["bandwidth_divisor"]),
            epsilon=configuration["epsilon"],
            n_samples=configuration["samples_per_batch"],
            **common,
        )
    if method == "thresholdppp":
        return ThresholdPPP(
            threshold=configuration["threshold"],
            method=configuration["method"],
            epsilon=configuration["epsilon"],
            n_samples=configuration["samples_per_batch"],
            **common,
        )
    raise ValueError(f"unknown benchmark method: {method}")


def _fit(
    estimator: Any,
    observations: np.ndarray,
    key: jax.Array,
    initial_covariance: np.ndarray | None,
) -> Any:
    if isinstance(estimator, (BMSPCov, SBMSPCov)) and initial_covariance is not None:
        fitted = estimator.fit(
            observations,
            key=key,
            initial_covariance=initial_covariance,
        )
    else:
        fitted = estimator.fit(observations, key=key)
    fitted.posterior_samples_packed_.block_until_ready()
    fitted.covariance_.block_until_ready()
    return fitted


def _fit_validity(estimator: Any) -> dict[str, object]:
    covariance = np.asarray(jax.device_get(estimator.covariance_), dtype=np.float64)
    finite = bool(np.all(np.isfinite(covariance)))
    symmetric = bool(np.allclose(covariance, covariance.T))
    spd = bool(finite and symmetric and np.all(np.linalg.eigvalsh(covariance) > 0.0))
    diagnostics = getattr(estimator, "diagnostics_", None)
    rejected = int(getattr(diagnostics, "n_rejected_sweeps", 0))
    return {
        "posterior_mean_finite": finite,
        "posterior_mean_symmetric": symmetric,
        "posterior_mean_spd": spd,
        "rejected_sweeps": rejected,
    }


def measure_cell(
    *,
    observations: np.ndarray,
    truth_covariance: np.ndarray,
    method: str,
    dtype: str,
    device: str,
    parallelism: int,
    manifest: Mapping[str, Any],
    seed: int,
    warm_repetitions: int = 3,
    initial_covariance: np.ndarray | None = None,
    smoke_samples: int | None = None,
) -> dict[str, object]:
    """Measure a cold fit and the adaptive warmed repetitions for one cell."""
    if warm_repetitions != 3:
        raise ValueError("the pre-registered protocol requires three initial repeats")
    dimension = observations.shape[1]

    def new_estimator() -> Any:
        estimator = build_estimator(
            method,
            dimension=dimension,
            dtype=dtype,
            device=device,
            parallelism=parallelism,
            manifest=manifest,
        )
        if smoke_samples is not None:
            estimator.n_samples = smoke_samples
            if hasattr(estimator, "burnin"):
                estimator.burnin = min(estimator.burnin, 1)
        return estimator

    keys = iter(jax.random.split(jax.random.key(seed), 6))
    cold_estimator = new_estimator()
    start = time.perf_counter()
    cold_estimator = _fit(
        cold_estimator, observations, next(keys), initial_covariance
    )
    cold_fit_seconds = time.perf_counter() - start

    warm_seconds: list[float] = []
    latest = cold_estimator
    for _ in range(3):
        latest = new_estimator()
        start = time.perf_counter()
        latest = _fit(latest, observations, next(keys), initial_covariance)
        warm_seconds.append(time.perf_counter() - start)
    threshold = float(manifest["timing"]["relative_range_threshold"])
    if needs_additional_repetitions(warm_seconds, threshold=threshold):
        for _ in range(2):
            latest = new_estimator()
            start = time.perf_counter()
            latest = _fit(latest, observations, next(keys), initial_covariance)
            warm_seconds.append(time.perf_counter() - start)

    retained_draws = int(latest.n_samples * latest.n_chains)
    truth = np.asarray(truth_covariance, dtype=np.float64)
    estimate = np.asarray(jax.device_get(latest.covariance_), dtype=np.float64)
    validity = _fit_validity(latest)
    validity["truth_relative_frobenius_error"] = float(
        np.linalg.norm(estimate - truth) / np.linalg.norm(truth)
    )
    return {
        "actual_platform": latest.device_.platform,
        "retained_draws": retained_draws,
        "cold_fit_seconds": cold_fit_seconds,
        "warm_seconds": warm_seconds,
        **validity,
    }


def _git_provenance(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, bool(status.strip())


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    project_root = script_directory.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=script_directory / "manifest.json")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("bm", "sbm", "bandppp", "thresholdppp"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--parallelism", type=int, required=True)
    parser.add_argument("--configuration", choices=("optimized", "cpu_baseline"), required=True)
    parser.add_argument("--cpu-cores", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    fixture = load_fixture(arguments.fixture_dir)
    metadata = json.loads(
        (arguments.fixture_dir / "metadata.json").read_text(encoding="utf-8")
    )
    result = measure_cell(
        observations=fixture["observations"],
        truth_covariance=fixture["truth_covariance"],
        initial_covariance=fixture["initial_covariance"],
        method=arguments.method,
        dtype=arguments.dtype,
        device=arguments.device,
        parallelism=arguments.parallelism,
        manifest=manifest,
        seed=int(manifest["seed"]),
    )
    revision, dirty = _git_provenance(project_root)
    record = {
        "schema_version": "1.0",
        "method": arguments.method,
        "dimension": int(metadata["dimension"]),
        "n_observations": int(metadata["n_observations"]),
        "seed": int(manifest["seed"]),
        "fixture_sha256": fixture_sha256(arguments.fixture_dir),
        "implementation": "pybspcov",
        "version": __version__,
        "device": arguments.device,
        "actual_platform": result.pop("actual_platform"),
        "dtype": arguments.dtype,
        "execution": "vmap" if arguments.parallelism > 1 else "single",
        "configuration": arguments.configuration,
        "parallelism": arguments.parallelism,
        "cpu_cores": arguments.cpu_cores,
        "cold_end_to_end_seconds": time.perf_counter() - PROCESS_START,
        "git_revision": revision,
        "git_dirty": dirty,
        "jax_version": jax.__version__,
        **result,
    }
    validate_timing_record(record)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
