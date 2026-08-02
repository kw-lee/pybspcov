import ast
import inspect

import jax
import jax.numpy as jnp
import pytest

import pybspcov.kernels.bm as bm_kernel
from pybspcov.kernels.bm import bm_sweep, initialize_bm_state
from pybspcov.sampling.gig import GIGSample


def _rollback_case() -> tuple[bm_kernel.BMState, tuple[jax.Array, ...]]:
    x = jnp.asarray(
        [
            [-1.0, 0.5, 0.2],
            [-0.4, -0.7, 0.1],
            [0.2, 0.1, -0.8],
            [0.5, -0.2, 0.6],
            [0.9, 0.4, -0.3],
            [-0.2, -0.1, 0.2],
        ],
        dtype=jnp.float32,
    )
    scatter = x.T @ x
    covariance = jnp.diag(jnp.diag(scatter) / x.shape[0])
    tau1sq = jnp.asarray(
        10_000.0 / (x.shape[0] * x.shape[1] ** 4),
        dtype=jnp.float32,
    )
    state = initialize_bm_state(covariance, tau1sq)
    arguments = (
        scatter,
        jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32),
        jnp.asarray(x.shape[0]),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        tau1sq,
    )
    return state, arguments


def _column_sampler_keys(key: jax.Array, column: int) -> tuple[jax.Array, jax.Array]:
    current_key = key
    for _ in range(column + 1):
        current_key, gamma_key, _, phi_key, _ = jax.random.split(current_key, 5)
    return gamma_key, phi_key


def _accepted_batch_gig(
    keys: jax.Array,
    lambda_: jax.Array,
    chi: jax.Array,
    psi: jax.Array,
) -> GIGSample:
    del keys, lambda_, psi
    return GIGSample(
        value=jnp.ones_like(chi),
        accepted=jnp.ones_like(chi, dtype=jnp.bool_),
        iterations=jnp.ones_like(chi, dtype=jnp.int32),
    )


def _assert_state_equal(actual: bm_kernel.BMState, expected: bm_kernel.BMState) -> None:
    for field_name, actual_value, expected_value in zip(
        bm_kernel.BMState._fields,
        actual,
        expected,
        strict=True,
    ):
        assert jnp.array_equal(actual_value, expected_value), field_name


def test_bm_sweep_routes_phi_updates_through_batched_gig() -> None:
    tree = ast.parse(inspect.getsource(bm_sweep))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_sample_gig_batch" in called_names


def test_bm_sweep_rolls_back_all_state_when_gamma_draw_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _rollback_case()
    sweep_key = jax.random.key(73)
    target_gamma_key, _ = _column_sampler_keys(sweep_key, 1)
    target_key_data = jax.random.key_data(target_gamma_key)

    def reject_target_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del lambda_, psi
        rejected = jnp.all(jax.random.key_data(key) == target_key_data)
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=~rejected,
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(bm_kernel, "sample_gig", reject_target_gamma)
    monkeypatch.setattr(bm_kernel, "_sample_gig_batch", _accepted_batch_gig)

    run_sweep = jax.jit(
        lambda key, initial_state, *sweep_arguments: bm_kernel.bm_sweep(
            key,
            initial_state,
            *sweep_arguments,
        )
    )
    result = run_sweep(sweep_key, state, *arguments)

    assert not result.accepted
    _assert_state_equal(result.state, state)


def test_bm_sweep_rolls_back_all_state_when_phi_draw_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _rollback_case()
    sweep_key = jax.random.key(79)

    def accept_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del key, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.asarray(True),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    def reject_target_phi(
        keys: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del keys, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.zeros_like(chi, dtype=jnp.bool_),
            iterations=jnp.ones_like(chi, dtype=jnp.int32),
        )

    monkeypatch.setattr(bm_kernel, "sample_gig", accept_gamma)
    monkeypatch.setattr(bm_kernel, "_sample_gig_batch", reject_target_phi)

    run_sweep = jax.jit(
        lambda key, initial_state, *sweep_arguments: bm_kernel.bm_sweep(
            key,
            initial_state,
            *sweep_arguments,
        )
    )
    result = run_sweep(sweep_key, state, *arguments)

    assert not result.accepted
    _assert_state_equal(result.state, state)


def test_sample_bm_chain_carries_input_state_through_rejected_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _rollback_case()

    def reject_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del key, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.asarray(False),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(bm_kernel, "sample_gig", reject_gamma)
    monkeypatch.setattr(bm_kernel, "_sample_gig_batch", _accepted_batch_gig)
    run_chain = jax.jit(
        lambda key, initial_state: bm_kernel.sample_bm_chain(
            key,
            initial_state,
            *arguments,
            burnin=1,
            n_samples=2,
        )
    )

    result = run_chain(jax.random.key(83), state)

    assert jnp.array_equal(result.accepted, jnp.asarray([False, False, False]))
    _assert_state_equal(result.final_state, state)
    assert jnp.array_equal(
        result.covariance,
        jnp.broadcast_to(state.covariance, result.covariance.shape),
    )
    assert jnp.array_equal(
        result.phi,
        jnp.broadcast_to(state.phi, result.phi.shape),
    )


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
