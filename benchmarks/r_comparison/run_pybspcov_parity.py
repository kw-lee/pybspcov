#!/usr/bin/env python3
"""Run one long pybspcov parity cell and emit its posterior summary."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
from core import load_manifest
from fixtures import fixture_sha256, load_fixture

from pybspcov import BandPPP, BMSPCov, SBMSPCov, ThresholdPPP, __version__


def build_parity_estimator(
    method: str,
    *,
    dtype: str,
    device: str,
    manifest: Mapping[str, Any],
) -> Any:
    """Construct the immutable long-run estimator used by the parity gate."""
    parity = manifest["parity"]
    common = {"dtype": dtype, "device": device}
    if method == "bm":
        return BMSPCov(
            burnin=parity["bm_sbm_burnin"],
            n_samples=parity["bm_sbm_samples_per_chain"],
            n_chains=parity["bm_sbm_chains"],
            **common,
        )
    if method == "sbm":
        return SBMSPCov(
            burnin=parity["bm_sbm_burnin"],
            n_samples=parity["bm_sbm_samples_per_chain"],
            n_chains=parity["bm_sbm_chains"],
            cutoff_method="correlation",
            retained_fraction=manifest["methods"]["sbm"]["retained_fraction"],
            screening_scope="chain",
            **common,
        )
    if method == "bandppp":
        return BandPPP(
            bandwidth=1,
            epsilon=manifest["methods"]["bandppp"]["epsilon"],
            n_samples=parity["ppp_total_samples"],
            n_chains=1,
            **common,
        )
    if method == "thresholdppp":
        configuration = manifest["methods"]["thresholdppp"]
        return ThresholdPPP(
            threshold=configuration["threshold"],
            method=configuration["method"],
            epsilon=configuration["epsilon"],
            n_samples=parity["ppp_total_samples"],
            n_chains=1,
            **common,
        )
    raise ValueError(f"unknown parity method: {method}")


def _batch_mcse(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def summarize_draws(
    draws: np.ndarray, truth: np.ndarray, *, n_batches: int
) -> dict[str, Any]:
    """Compute the pre-registered posterior statistics and batch MCSEs."""
    draws = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != truth.shape:
        raise ValueError("draws and truth have incompatible covariance shapes")
    if n_batches < 2 or draws.shape[0] < 2 * n_batches:
        raise ValueError("parity summaries require at least two draws per batch")
    batch_size = draws.shape[0] // n_batches
    trimmed = draws[: n_batches * batch_size]
    batches = trimmed.reshape(n_batches, batch_size, *truth.shape)
    probabilities = (0.025, 0.5, 0.975)
    posterior_mean = np.mean(draws, axis=0)
    batch_means = np.mean(batches, axis=1)
    quantiles = np.quantile(draws, probabilities, axis=0, method="linear")
    batch_quantiles = np.stack(
        [
            np.quantile(batch, probabilities, axis=0, method="linear")
            for batch in batches
        ]
    )
    batch_rmses = np.sqrt(np.mean(np.square(batch_means - truth), axis=(1, 2)))
    return {
        "truth": truth,
        "posterior_mean": posterior_mean,
        "posterior_mean_mcse": _batch_mcse(batch_means),
        "posterior_sd": np.std(draws, axis=0, ddof=1),
        "posterior_sd_mcse": _batch_mcse(np.std(batches, axis=1, ddof=1)),
        "q025": quantiles[0],
        "q025_mcse": _batch_mcse(batch_quantiles[:, 0]),
        "q50": quantiles[1],
        "q50_mcse": _batch_mcse(batch_quantiles[:, 1]),
        "q975": quantiles[2],
        "q975_mcse": _batch_mcse(batch_quantiles[:, 2]),
        "rmse": float(np.sqrt(np.mean(np.square(posterior_mean - truth)))),
        "rmse_mcse": float(_batch_mcse(batch_rmses)),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "trimmed_samples": int(trimmed.shape[0]),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


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
    ).stdout.strip()
    return revision, bool(status)


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    project_root = script_directory.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=script_directory / "manifest.json"
    )
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("bm", "sbm", "bandppp", "thresholdppp"), required=True
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    fixture = load_fixture(arguments.fixture_dir)
    estimator = build_parity_estimator(
        arguments.method,
        dtype=arguments.dtype,
        device="cpu",
        manifest=manifest,
    )
    fit_arguments: dict[str, Any] = {"key": jax.random.key(int(manifest["seed"]))}
    if isinstance(estimator, (BMSPCov, SBMSPCov)):
        fit_arguments["initial_covariance"] = fixture["initial_covariance"]
    estimator.fit(fixture["observations"], **fit_arguments)
    samples = np.asarray(jax.device_get(estimator.posterior_samples_), dtype=np.float64)
    samples = samples.reshape((-1, samples.shape[-2], samples.shape[-1]))
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("parity sampler produced nonfinite covariance draws")
    diagnostics = getattr(estimator, "diagnostics_", None)
    rejected = int(getattr(diagnostics, "n_rejected_sweeps", 0))
    if rejected:
        raise RuntimeError(f"parity sampler rejected {rejected} sweep(s)")
    revision, dirty = _git_provenance(project_root)
    if dirty:
        raise RuntimeError("parity artifacts require a clean git worktree")
    record = {
        "schema_version": "1.0",
        "method": arguments.method,
        "implementation": "pybspcov",
        "version": __version__,
        "dtype": arguments.dtype,
        "device": "cpu",
        "fixture_sha256": fixture_sha256(arguments.fixture_dir),
        "git_revision": revision,
        "git_dirty": dirty,
        "summary": summarize_draws(
            samples,
            fixture["truth_covariance"],
            n_batches=int(manifest["parity"]["batches"]),
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(_jsonable(record), sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
