"""Generate identical versioned inputs for the R and Python runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Fixture = dict[str, FloatArray]
FILENAMES = {
    "observations": "observations.csv",
    "truth_covariance": "truth_covariance.csv",
    "initial_covariance": "initial_covariance.csv",
}


def _sparse_covariance(
    dimension: int, density: float, rng: np.random.Generator
) -> FloatArray:
    lower_rows, lower_columns = np.tril_indices(dimension, k=-1)
    edge_count = round(density * lower_rows.size)
    selected = rng.choice(lower_rows.size, size=edge_count, replace=False)
    weights = rng.uniform(0.05, 0.2, size=edge_count)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=edge_count)
    precision = np.zeros((dimension, dimension), dtype=np.float64)
    precision[lower_rows[selected], lower_columns[selected]] = weights * signs
    precision[lower_columns[selected], lower_rows[selected]] = weights * signs
    np.fill_diagonal(precision, np.sum(np.abs(precision), axis=1) + 1.0)
    return np.asarray(np.linalg.inv(precision), dtype=np.float64)


def _banded_covariance(dimension: int, bandwidth: int) -> FloatArray:
    indices = np.arange(dimension)
    distance = np.abs(indices[:, None] - indices[None, :])
    covariance = np.where(distance <= bandwidth, 0.3**distance, 0.0)
    return np.asarray(covariance, dtype=np.float64)


def generate_fixture(
    *,
    dimension: int,
    n_observations: int,
    seed: int,
    kind: Literal["sparse", "banded"],
    density: float | None = None,
    bandwidth: int | None = None,
) -> Fixture:
    """Generate a deterministic centered Gaussian benchmark fixture."""
    if dimension < 2 or n_observations < 2:
        raise ValueError("dimension and n_observations must be at least two")
    rng = np.random.default_rng(seed)
    if kind == "sparse":
        if density is None or not 0.0 <= density <= 1.0:
            raise ValueError("sparse fixtures require density between zero and one")
        truth = _sparse_covariance(dimension, density, rng)
    elif kind == "banded":
        if bandwidth is None or not 1 <= bandwidth < dimension:
            raise ValueError("banded fixtures require 1 <= bandwidth < dimension")
        truth = _banded_covariance(dimension, bandwidth)
    else:
        raise ValueError("kind must be 'sparse' or 'banded'")

    standard = rng.normal(size=(n_observations, dimension))
    observations = standard @ np.linalg.cholesky(truth).T
    observations -= observations.mean(axis=0, keepdims=True)
    initial = np.diag(np.diag(np.cov(observations, rowvar=False, ddof=1)))
    return {
        "observations": np.ascontiguousarray(observations, dtype=np.float64),
        "truth_covariance": np.ascontiguousarray(truth, dtype=np.float64),
        "initial_covariance": np.ascontiguousarray(initial, dtype=np.float64),
    }


def fixture_sha256(directory: Path) -> str:
    """Hash the exact three CSV payloads consumed by both implementations."""
    digest = hashlib.sha256()
    for key, filename in FILENAMES.items():
        digest.update(key.encode("ascii"))
        digest.update(b"\0")
        digest.update((directory / filename).read_bytes())
    return digest.hexdigest()


def write_fixture(directory: Path, fixture: Fixture) -> dict[str, object]:
    """Write one fixture using round-trippable CSV and return its metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    shapes = {np.asarray(value).shape for value in fixture.values()}
    observations = np.asarray(fixture["observations"], dtype=np.float64)
    dimension = observations.shape[1]
    if shapes != {
        observations.shape,
        (dimension, dimension),
    }:
        raise ValueError("fixture matrices have incompatible shapes")
    for key, filename in FILENAMES.items():
        np.savetxt(directory / filename, fixture[key], delimiter=",", fmt="%.17g")
    metadata = {
        "dimension": dimension,
        "n_observations": observations.shape[0],
        "sha256": fixture_sha256(directory),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_fixture(directory: Path) -> Fixture:
    """Load the exact matrices written by :func:`write_fixture`."""
    return {
        key: np.loadtxt(directory / filename, delimiter=",", dtype=np.float64)
        for key, filename in FILENAMES.items()
    }
