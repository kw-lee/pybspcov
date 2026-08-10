import jax
import jax.numpy as jnp

from pybspcov.kernels.covariance import update_covariance_column


def test_column_update_keeps_covariance_and_precision_consistent() -> None:
    covariance = jnp.array(
        [[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.0]],
        dtype=jnp.float64,
    )
    precision = jnp.linalg.inv(covariance)
    other_indices = jnp.array([0, 2])
    beta = jnp.array([0.3, -0.2], dtype=jnp.float64)

    updated_covariance, updated_precision = jax.jit(update_covariance_column)(
        covariance,
        precision,
        jnp.array(1),
        other_indices,
        beta,
        jnp.array(0.8),
    )

    assert jnp.allclose(updated_covariance, updated_covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(updated_covariance) > 0.0)
    assert jnp.allclose(
        updated_precision,
        jnp.linalg.inv(updated_covariance),
        rtol=1e-12,
        atol=1e-12,
    )
