import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pybspcov import ThresholdPPP

FIXTURE_DIR = Path("tests/fixtures/r/bspcov-1.0.3")


def test_thresholdppp_mean_matches_r_with_combined_monte_carlo_error() -> None:
    x = np.loadtxt(FIXTURE_DIR / "thresholdppp_x.csv", delimiter=",")
    with (FIXTURE_DIR / "thresholdppp_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    r_mean = np.asarray([float(row["mean"]) for row in summary])
    r_mcse = np.asarray([float(row["mean_mcse"]) for row in summary])

    model = ThresholdPPP(
        threshold=0.4,
        method="hard",
        epsilon=0.05,
        n_samples=5_000,
        dtype="float32",
    ).fit(jnp.asarray(x, dtype=jnp.float32), key=jax.random.key(20260820))

    draws = np.asarray(model.posterior_samples_packed_[0])
    batch_means = draws.reshape(50, 100, draws.shape[-1]).mean(axis=1)
    python_mean = draws.mean(axis=0)
    python_mcse = batch_means.std(axis=0, ddof=1) / np.sqrt(50.0)
    combined_mcse = np.sqrt(r_mcse**2 + python_mcse**2)

    assert np.all(np.abs(python_mean - r_mean) <= 6.0 * combined_mcse)
