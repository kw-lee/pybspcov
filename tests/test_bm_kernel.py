import jax
import jax.numpy as jnp

from pybspcov.kernels.bm import bm_sweep, initialize_bm_state


def test_bm_sweep_preserves_state_invariants_and_is_reproducible() -> None:
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

    run_sweep = jax.jit(bm_sweep)
    result = run_sweep(
        jax.random.key(41),
        state,
        scatter,
        other_indices,
        jnp.array(x.shape[0]),
        jnp.array(0.5),
        jnp.array(0.5),
        jnp.array(1.0),
        tau1sq,
    )
    repeated = run_sweep(
        jax.random.key(41),
        state,
        scatter,
        other_indices,
        jnp.array(x.shape[0]),
        jnp.array(0.5),
        jnp.array(0.5),
        jnp.array(1.0),
        tau1sq,
    )

    assert result.accepted
    assert jnp.all(jnp.isfinite(result.state.covariance))
    assert jnp.allclose(result.state.covariance, result.state.covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(result.state.covariance) > 0.0)
    assert jnp.allclose(
        result.state.precision,
        jnp.linalg.inv(result.state.covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    assert jnp.allclose(result.state.phi, result.state.phi.T)
    assert jnp.allclose(result.state.psi, result.state.psi.T)
    assert jnp.allclose(result.state.tau, result.state.tau.T)
    assert jnp.allclose(result.state.covariance, repeated.state.covariance)
    assert jnp.allclose(result.state.phi, repeated.state.phi)
