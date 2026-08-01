from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pybspcov.kernels.covariance import update_covariance_column

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_csv(name: str, *, dtype: type[np.floating] = np.float64) -> np.ndarray:
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", dtype=dtype, ndmin=1)


def test_covariance_column_update_matches_bspcov_1_0_3() -> None:
    parameters = _load_csv("parameters.csv")
    covariance = jnp.asarray(_load_csv("covariance.csv"), dtype=jnp.float64)
    precision = jnp.asarray(_load_csv("precision.csv"), dtype=jnp.float64)
    beta = jnp.asarray(_load_csv("beta.csv"), dtype=jnp.float64)
    other_indices = jnp.asarray(
        _load_csv("other_indices.csv", dtype=np.int64), dtype=jnp.int32
    )

    updated_covariance, updated_precision = jax.jit(update_covariance_column)(
        covariance,
        precision,
        jnp.asarray(int(parameters[0]), dtype=jnp.int32),
        other_indices,
        beta,
        jnp.asarray(parameters[1], dtype=jnp.float64),
    )

    assert updated_covariance.dtype == jnp.float64
    assert updated_precision.dtype == jnp.float64
    np.testing.assert_allclose(
        np.asarray(updated_covariance),
        _load_csv("expected_covariance.csv"),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(updated_precision),
        _load_csv("expected_precision.csv"),
        rtol=1e-12,
        atol=1e-12,
    )
