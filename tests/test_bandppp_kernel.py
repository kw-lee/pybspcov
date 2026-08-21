import jax
import jax.numpy as jnp
import pytest

from pybspcov.kernels.bandppp import (
    band_and_adjust_covariance,
    sample_inverse_wishart,
)


def test_band_and_adjust_covariance_enforces_band_and_eigenvalue_floor() -> None:
    covariance = jnp.asarray(
        [
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ],
        dtype=jnp.float32,
    )

    adjusted, was_adjusted = band_and_adjust_covariance(
        covariance,
        bandwidth=1,
        epsilon=jnp.asarray(0.1, dtype=jnp.float32),
    )

    expected_diagonal = 1.23137085
    expected = jnp.asarray(
        [
            [expected_diagonal, 0.8, 0.0],
            [0.8, expected_diagonal, 0.8],
            [0.0, 0.8, expected_diagonal],
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(adjusted, expected, rtol=1e-6, atol=1e-6)
    assert bool(was_adjusted)
    assert float(jnp.linalg.eigvalsh(adjusted)[0]) == pytest.approx(
        0.1,
        abs=2e-6,
    )


def test_inverse_wishart_draws_match_the_analytic_mean() -> None:
    scale = jnp.asarray([[2.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)
    degrees_of_freedom = jnp.asarray(12.0, dtype=jnp.float32)

    draws = sample_inverse_wishart(
        jax.random.key(20260820),
        degrees_of_freedom=degrees_of_freedom,
        scale=scale,
        n_samples=40_000,
    )

    expected = scale / 9.0
    assert draws.shape == (40_000, 2, 2)
    assert jnp.allclose(jnp.mean(draws, axis=0), expected, rtol=0.025, atol=0.003)
