import jax.numpy as jnp

from pybspcov.kernels.thresholdppp import threshold_and_adjust_covariance


def test_hard_threshold_preserves_diagonal_and_removes_small_edges() -> None:
    covariance = jnp.asarray(
        [[1.0, 0.2, -0.6], [0.2, 2.0, 0.4], [-0.6, 0.4, 3.0]],
        dtype=jnp.float32,
    )

    adjusted, was_adjusted = threshold_and_adjust_covariance(
        covariance,
        threshold=jnp.asarray(0.4, dtype=jnp.float32),
        method="hard",
        epsilon=jnp.asarray(0.0, dtype=jnp.float32),
    )

    expected = jnp.asarray(
        [[1.0, 0.0, -0.6], [0.0, 2.0, 0.4], [-0.6, 0.4, 3.0]],
        dtype=jnp.float32,
    )
    assert jnp.allclose(adjusted, expected)
    assert not bool(was_adjusted)


def test_soft_threshold_shrinks_only_off_diagonal_entries() -> None:
    covariance = jnp.asarray(
        [[1.0, 0.2, -0.6], [0.2, 2.0, 0.4], [-0.6, 0.4, 3.0]],
        dtype=jnp.float32,
    )

    adjusted, _ = threshold_and_adjust_covariance(
        covariance,
        threshold=jnp.asarray(0.25, dtype=jnp.float32),
        method="soft",
        epsilon=jnp.asarray(0.0, dtype=jnp.float32),
    )

    expected = jnp.asarray(
        [[1.0, 0.0, -0.35], [0.0, 2.0, 0.15], [-0.35, 0.15, 3.0]],
        dtype=jnp.float32,
    )
    assert jnp.allclose(adjusted, expected)


def test_threshold_adjustment_enforces_requested_eigenvalue_floor() -> None:
    covariance = jnp.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float32)

    adjusted, was_adjusted = threshold_and_adjust_covariance(
        covariance,
        threshold=jnp.asarray(0.0, dtype=jnp.float32),
        method="hard",
        epsilon=jnp.asarray(0.2, dtype=jnp.float32),
    )

    assert bool(was_adjusted)
    assert float(jnp.linalg.eigvalsh(adjusted)[0]) >= 0.2 - 1e-6
