import jax
import jax.numpy as jnp
import pytest

from pybspcov.kernels.bm import initialize_bm_state, sample_bm_chain


def test_sample_bm_chain_discards_burnin_and_is_reproducible() -> None:
    x = jnp.array(
        [
            [-1.0, 0.5, 0.2],
            [-0.4, -0.7, 0.1],
            [0.2, 0.1, -0.8],
            [0.5, -0.2, 0.6],
            [0.9, 0.4, -0.3],
            [-0.2, -0.1, 0.2],
        ],
        dtype=jnp.float64,
    )
    scatter = x.T @ x
    covariance = jnp.diag(jnp.diag(scatter) / x.shape[0])
    tau1sq = jnp.array(10_000.0 / (x.shape[0] * x.shape[1] ** 4))
    state = initialize_bm_state(covariance, tau1sq)
    other_indices = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    run_chain = jax.jit(
        sample_bm_chain,
        static_argnames=("burnin", "n_samples"),
    )

    result = run_chain(
        jax.random.key(42),
        state,
        scatter,
        other_indices,
        jnp.array(x.shape[0]),
        jnp.array(0.5),
        jnp.array(0.5),
        jnp.array(1.0),
        tau1sq,
        burnin=2,
        n_samples=3,
    )
    repeated = run_chain(
        jax.random.key(42),
        state,
        scatter,
        other_indices,
        jnp.array(x.shape[0]),
        jnp.array(0.5),
        jnp.array(0.5),
        jnp.array(1.0),
        tau1sq,
        burnin=2,
        n_samples=3,
    )

    assert result.covariance.shape == (3, 3, 3)
    assert result.phi.shape == (3, 3, 3)
    assert result.accepted.shape == (5,)
    assert jnp.all(result.accepted)
    assert jnp.allclose(result.final_state.covariance, result.covariance[-1])
    assert jnp.allclose(result.covariance, repeated.covariance)
    assert jnp.allclose(result.phi, repeated.phi)


@pytest.mark.parametrize(
    ("burnin", "n_samples", "message"),
    [
        (-1, 1, "burnin must be non-negative"),
        (0, 0, "n_samples must be positive"),
    ],
)
def test_sample_bm_chain_validates_static_lengths(
    burnin: int,
    n_samples: int,
    message: str,
) -> None:
    covariance = jnp.eye(2, dtype=jnp.float64)
    state = initialize_bm_state(covariance, jnp.array(1.0, dtype=jnp.float64))

    with pytest.raises(ValueError, match=message):
        sample_bm_chain(
            jax.random.key(0),
            state,
            covariance,
            jnp.array([[1], [0]], dtype=jnp.int32),
            jnp.array(2),
            jnp.array(0.5),
            jnp.array(0.5),
            jnp.array(1.0),
            jnp.array(1.0),
            burnin=burnin,
            n_samples=n_samples,
        )
