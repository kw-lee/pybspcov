import csv
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pybspcov.sampling.gig import GIGSample, sample_gig

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "r"
    / "GIGrvg-0.8"
    / "gig_summary.csv"
)
JAX_SAMPLE_COUNT = 32_768
MONTE_CARLO_STANDARD_ERRORS = 6.0


def _load_summaries() -> list[dict[str, str]]:
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as fixture_file:
        return list(csv.DictReader(fixture_file))


@jax.jit
def _sample_many(
    keys: jax.Array,
    lambda_: jax.Array,
    chi: jax.Array,
    psi: jax.Array,
) -> GIGSample:
    return jax.vmap(sample_gig, in_axes=(0, None, None, None))(
        keys, lambda_, chi, psi
    )


@pytest.mark.parametrize(
    ("summary", "stream"),
    [(summary, stream) for stream, summary in enumerate(_load_summaries())],
    ids=lambda value: value["regime"] if isinstance(value, dict) else None,
)
def test_gig_sample_mean_matches_gigrvg(
    summary: dict[str, str], stream: int
) -> None:
    assert jax.config.x64_enabled
    keys = jax.random.split(
        jax.random.fold_in(jax.random.key(20260801), stream), JAX_SAMPLE_COUNT
    )
    samples = _sample_many(
        keys,
        jnp.asarray(float(summary["lambda"]), dtype=jnp.float64),
        jnp.asarray(float(summary["chi"]), dtype=jnp.float64),
        jnp.asarray(float(summary["psi"]), dtype=jnp.float64),
    )

    accepted = np.asarray(samples.accepted)
    assert np.all(accepted), (
        f"{summary['regime']}: {accepted.size - np.count_nonzero(accepted)} "
        "draws were rejected"
    )
    values = np.asarray(samples.value)
    assert np.all(np.isfinite(values))

    jax_standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    combined_standard_error = math.hypot(
        float(summary["standard_error"]), jax_standard_error
    )
    tolerance = MONTE_CARLO_STANDARD_ERRORS * combined_standard_error
    np.testing.assert_allclose(
        np.mean(values),
        float(summary["mean"]),
        rtol=0.0,
        atol=tolerance,
        err_msg=(
            f"{summary['regime']}: means differ by more than "
            f"{MONTE_CARLO_STANDARD_ERRORS:g} combined standard errors"
        ),
    )
