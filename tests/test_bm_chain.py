import jax
import jax.numpy as jnp
import pytest

import pybspcov.kernels.bm as bm_kernel
from pybspcov.kernels import sample_bm_chains
from pybspcov.kernels.bm import bm_sweep, initialize_bm_state, sample_bm_chain
from pybspcov.sampling.gig import GIGSample


def _bm_chain_case(
    dtype: jnp.dtype,
) -> tuple[bm_kernel.BMState, tuple[jax.Array, ...]]:
    x = jnp.asarray(
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
    scatter = x.T @ x
    covariance = jnp.diag(jnp.diag(scatter) / x.shape[0])
    tau1sq = jnp.asarray(
        10_000.0 / (x.shape[0] * x.shape[1] ** 4),
        dtype=dtype,
    )
    state = initialize_bm_state(covariance, tau1sq)
    arguments = (
        scatter,
        jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32),
        jnp.asarray(x.shape[0]),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    )
    return state, arguments


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
    master_key = jax.random.key(42)
    run_chain = jax.jit(
        sample_bm_chain,
        static_argnames=("burnin", "n_samples"),
    )

    result = run_chain(
        master_key,
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
        master_key,
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

    expected_state = state
    expected_covariance = []
    expected_phi = []
    expected_accepted = []
    for sweep_key in jax.random.split(master_key, 5):
        sweep_result = bm_sweep(
            sweep_key,
            expected_state,
            scatter,
            other_indices,
            jnp.array(x.shape[0]),
            jnp.array(0.5),
            jnp.array(0.5),
            jnp.array(1.0),
            tau1sq,
        )
        expected_state = sweep_result.state
        expected_covariance.append(expected_state.covariance)
        expected_phi.append(expected_state.phi)
        expected_accepted.append(sweep_result.accepted)

    expected_covariance_array = jnp.stack(expected_covariance[2:])
    expected_phi_array = jnp.stack(expected_phi[2:])
    expected_accepted_array = jnp.stack(expected_accepted)
    assert result.covariance.shape == (3, 3, 3)
    assert result.phi.shape == (3, 3, 3)
    assert result.accepted.shape == (5,)
    assert jnp.all(result.accepted)
    assert jnp.allclose(result.covariance, expected_covariance_array)
    assert jnp.allclose(result.phi, expected_phi_array)
    assert jnp.array_equal(result.accepted, expected_accepted_array)
    for actual, expected in zip(result.final_state, expected_state, strict=True):
        assert jnp.allclose(actual, expected)
    assert jnp.allclose(result.covariance, repeated.covariance)
    assert jnp.allclose(result.phi, repeated.phi)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sample_bm_chains_matches_independent_single_chain_calls(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    state, arguments = _bm_chain_case(dtype)
    states = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (3, *value.shape)),
        state,
    )
    keys = jax.random.split(jax.random.key(61), 3)
    run_chains = jax.jit(
        sample_bm_chains,
        static_argnames=("burnin", "n_samples"),
    )

    result = run_chains(
        keys,
        states,
        *arguments,
        burnin=1,
        n_samples=3,
    )
    independent = [
        sample_bm_chain(
            key,
            state,
            *arguments,
            burnin=1,
            n_samples=3,
        )
        for key in keys
    ]

    assert result.covariance.shape == (3, 3, 3, 3)
    assert result.phi.shape == (3, 3, 3, 3)
    assert result.accepted.shape == (3, 4)
    assert result.covariance.dtype == dtype
    assert result.phi.dtype == dtype
    assert jnp.allclose(
        result.covariance,
        jnp.stack([chain.covariance for chain in independent]),
    )
    assert jnp.allclose(result.phi, jnp.stack([chain.phi for chain in independent]))
    assert jnp.array_equal(
        result.accepted,
        jnp.stack([chain.accepted for chain in independent]),
    )
    for index, actual in enumerate(result.final_state):
        assert actual.dtype == dtype
        assert jnp.allclose(
            actual,
            jnp.stack([chain.final_state[index] for chain in independent]),
        )
    assert not jnp.allclose(result.covariance[0], result.covariance[1])


def test_sample_bm_chains_preserves_each_state_through_rejected_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _bm_chain_case(jnp.float32)
    second_state = initialize_bm_state(
        state.covariance * jnp.asarray(1.25, dtype=jnp.float32),
        arguments[-1],
    )
    states = jax.tree.map(
        lambda left, right: jnp.stack([left, right]),
        state,
        second_state,
    )

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

    def accept_phi(
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

    monkeypatch.setattr(bm_kernel, "sample_gig", reject_gamma)
    monkeypatch.setattr(bm_kernel, "_sample_gig_batch", accept_phi)
    run_chains = jax.jit(
        sample_bm_chains,
        static_argnames=("burnin", "n_samples"),
    )

    result = run_chains(
        jax.random.split(jax.random.key(83), 2),
        states,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    assert jnp.array_equal(result.accepted, jnp.zeros((2, 3), dtype=jnp.bool_))
    for actual, expected in zip(result.final_state, states, strict=True):
        assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(
        result.covariance,
        jnp.broadcast_to(states.covariance[:, None], result.covariance.shape),
    )
    assert jnp.array_equal(
        result.phi,
        jnp.broadcast_to(states.phi[:, None], result.phi.shape),
    )


def test_sample_bm_chains_validates_chain_axis() -> None:
    state, arguments = _bm_chain_case(jnp.float32)
    states = jax.tree.map(lambda value: jnp.stack([value, value]), state)

    with pytest.raises(TypeError, match="typed JAX keys"):
        sample_bm_chains(
            jax.random.PRNGKey(65),
            states,
            *arguments,
            burnin=0,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="one-dimensional batch"):
        sample_bm_chains(
            jax.random.PRNGKey(67)[None],
            states,
            *arguments,
            burnin=0,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="at least one chain"):
        sample_bm_chains(
            jax.random.split(jax.random.key(71), 0),
            jax.tree.map(lambda value: value[:0], states),
            *arguments,
            burnin=0,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="state leading dimension"):
        sample_bm_chains(
            jax.random.split(jax.random.key(73), 3),
            states,
            *arguments,
            burnin=0,
            n_samples=1,
        )


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
