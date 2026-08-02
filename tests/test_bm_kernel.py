import jax
import jax.numpy as jnp
import pytest

from pybspcov.kernels.bm import bm_sweep, initialize_bm_state


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_bm_sweep_preserves_state_invariants_and_is_reproducible(
    dtype_name: str,
) -> None:
    dtype = getattr(jnp, dtype_name)
    tolerance = 1e-5 if dtype_name == "float32" else 1e-10
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    x = jnp.array(
        [
            [-1.0, 0.5, 0.2],
            [-0.4, -0.7, 0.1],
            [0.2, 0.1, -0.8],
            [0.5, -0.2, 0.6],
            [0.9, 0.4, -0.3],
            [-0.2, -0.1, 0.2],
        ],
        dtype=dtype,
    )
    assert x.dtype == dtype
    scatter = x.T @ x
    covariance = jnp.diag(jnp.diag(scatter) / x.shape[0])
    tau1sq = jnp.array(10_000.0 / (x.shape[0] * x.shape[1] ** 4), dtype=dtype)
    state = initialize_bm_state(covariance, tau1sq)
    other_indices = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)

    run_sweep = jax.jit(bm_sweep)
    result = run_sweep(
        jax.random.key(41),
        state,
        scatter,
        other_indices,
        jnp.array(x.shape[0]),
        jnp.array(0.5, dtype=dtype),
        jnp.array(0.5, dtype=dtype),
        jnp.array(1.0, dtype=dtype),
        tau1sq,
    )
    repeated = run_sweep(
        jax.random.key(41),
        state,
        scatter,
        other_indices,
        jnp.array(x.shape[0]),
        jnp.array(0.5, dtype=dtype),
        jnp.array(0.5, dtype=dtype),
        jnp.array(1.0, dtype=dtype),
        tau1sq,
    )

    assert result.accepted
    assert jnp.all(jnp.isfinite(result.state.covariance))
    assert jnp.allclose(result.state.covariance, result.state.covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(result.state.covariance) > 0.0)
    assert jnp.allclose(
        result.state.precision,
        jnp.linalg.inv(result.state.covariance),
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(result.state.phi, result.state.phi.T)
    assert jnp.allclose(result.state.psi, result.state.psi.T)
    assert jnp.allclose(result.state.tau, result.state.tau.T)
    assert jnp.allclose(result.state.covariance, repeated.state.covariance)
    assert jnp.allclose(result.state.phi, repeated.state.phi)
