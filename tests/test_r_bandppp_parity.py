import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pybspcov import BandPPP

FIXTURE_DIR = Path("tests/fixtures/r/bspcov-1.0.3")


def _read_matrix(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",")


def test_bandppp_posterior_mean_matches_r_with_combined_monte_carlo_error() -> None:
    x = _read_matrix("bandppp_x.csv")
    with (FIXTURE_DIR / "bandppp_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    r_mean = np.asarray([float(row["mean"]) for row in summary])
    r_mcse = np.asarray([float(row["mean_mcse"]) for row in summary])

    model = BandPPP(
        bandwidth=1,
        epsilon=0.05,
        n_samples=5_000,
        dtype="float32",
    ).fit(jnp.asarray(x, dtype=jnp.float32), key=jax.random.key(20260820))

    draws = np.asarray(model.posterior_samples_packed_[0])
    batch_means = draws.reshape(50, 100, draws.shape[-1]).mean(axis=1)
    python_mean = draws.mean(axis=0)
    python_mcse = batch_means.std(axis=0, ddof=1) / np.sqrt(50.0)
    active = np.asarray([True, True, False, True, True, True])
    combined_mcse = np.sqrt(r_mcse**2 + python_mcse**2)

    assert np.array_equal(python_mean[~active], r_mean[~active])
    assert np.all(
        np.abs(python_mean[active] - r_mean[active]) <= 6.0 * combined_mcse[active]
    )
