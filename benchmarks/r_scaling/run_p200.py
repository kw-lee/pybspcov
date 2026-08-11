"""Run repeated p=200 BM fits with R bspcov on a shared fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _generate_fixture(**kwargs: object) -> Any:
    project_root = Path(__file__).resolve().parents[2]
    scaling_path = project_root / "benchmarks" / "sbm_public_scaling.py"
    spec = importlib.util.spec_from_file_location(
        "pybspcov_scaling_fixture",
        scaling_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scaling fixture generator from {scaling_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_fixture(**kwargs)


def write_shared_fixture(
    directory: Path,
    *,
    dimension: int,
    density: float,
    n_observations: int,
    seed: int,
) -> dict[str, object]:
    """Write one canonical float64 fixture for R and Python comparisons."""
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
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / "observations.csv", observations, delimiter=",", fmt="%.17g")
    np.savetxt(directory / "truth_covariance.csv", truth, delimiter=",", fmt="%.17g")
    np.savetxt(
        directory / "initial_covariance.csv",
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


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("normalized timings must be finite positive values")
    q1, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    return {
        "median": float(statistics.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_r_result(
    repetition_records: Sequence[Mapping[str, object]],
    *,
    fixture: Mapping[str, object],
    burnin: int,
    n_samples: int,
    chain_count: int,
    r_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate repeated R measurements and build an R-only result document."""
    if not repetition_records:
        raise ValueError("at least one R repetition is required")
    expected_draws = n_samples * chain_count
    normalized_timings: list[float] = []
    validated: list[dict[str, object]] = []
    for expected_index, source in enumerate(repetition_records):
        record = dict(source)
        if int(record.get("repetition", -1)) != expected_index:
            raise ValueError("R repetition indices must be contiguous from zero")
        if int(record.get("retained_draws", -1)) != expected_draws:
            raise ValueError(
                f"each R repetition must contain {expected_draws} retained draws"
            )
        if not all(
            record.get(field) is True
            for field in (
                "posterior_mean_finite",
                "posterior_mean_symmetric",
                "posterior_mean_spd",
            )
        ):
            raise ValueError("each R repetition must have a valid posterior mean")
        total = float(record["total_wall_seconds"])
        normalized = float(record["normalized_wall_seconds_per_chain"])
        if not np.isclose(normalized, total / chain_count, rtol=1e-12, atol=0.0):
            raise ValueError(
                "R normalized wall time must equal total wall time / chains"
            )
        if not np.isclose(
            float(record["chains_per_second"]),
            1.0 / normalized,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("R throughput must be the reciprocal normalized time")
        normalized_timings.append(normalized)
        validated.append(record)

    r_result: dict[str, object] = {
        "implementation": "bspcov",
        "device": "cpu",
        "dtype": "float64",
        "execution_model": "parallel",
    }
    if r_metadata is not None:
        r_result.update(dict(r_metadata))
    r_result.update(
        measured_repetitions=validated,
        timing_summary=_distribution_summary(normalized_timings),
    )
    return {
        "benchmark": "p200-r-bm-comparison",
        "schema_version": "2.0",
        "fixture": dict(fixture),
        "configuration": {
            "burnin": burnin,
            "n_samples": n_samples,
            "chain_count": chain_count,
            "repetitions": len(validated),
        },
        "r": r_result,
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
    repetitions: int,
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
            "--repetitions",
            str(repetitions),
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
        raise TypeError("R runner output must be a JSON object")
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
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260803)
    arguments = parser.parse_args(argv)
    if arguments.dimension < 2 or arguments.n_factor < 1:
        parser.error("dimension must be at least two and n-factor must be positive")
    if (
        arguments.burnin < 0
        or arguments.n_samples < 1
        or arguments.n_chains < 1
        or arguments.repetitions < 1
    ):
        parser.error(
            "burnin must be non-negative; samples, chains, and repetitions must be positive"
        )
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
        raw = _run_r(
            fixture_directory,
            r_library=arguments.r_library,
            fixture_sha256=str(fixture["fixture_sha256"]),
            burnin=arguments.burnin,
            n_samples=arguments.n_samples,
            chain_count=arguments.n_chains,
            repetitions=arguments.repetitions,
            seed=arguments.seed,
        )

    metadata = raw.get("metadata")
    repetitions = raw.get("measured_repetitions")
    if not isinstance(metadata, dict) or not isinstance(repetitions, list):
        raise TypeError("R runner must return metadata and measured_repetitions")
    output = build_r_result(
        repetitions,
        fixture=fixture,
        burnin=arguments.burnin,
        n_samples=arguments.n_samples,
        chain_count=arguments.n_chains,
        r_metadata=metadata,
    )
    output["git"] = _git_provenance(project_root)
    output["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
