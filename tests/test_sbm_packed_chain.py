from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pybspcov import kernels
from pybspcov.kernels import sbm
from pybspcov.kernels.bm import unpack_lower_triangle_column_major
from pybspcov.sampling.gig import GIGSample

ACTIVE_MASK = jnp.asarray(
    [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]
)
OTHER_INDICES = jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)


def _compact_chain_case(
    dtype: jnp.dtype,
    *,
    active_mask: jax.Array = ACTIVE_MASK,
) -> tuple[sbm.BMState, tuple[object, ...]]:
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
    covariance = jnp.asarray(
        [[1.5, 0.1, 0.0], [0.1, 1.2, -0.08], [0.0, -0.08, 1.1]],
        dtype=dtype,
    )
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, active_mask)
    arguments = (
        x.T @ x,
        jnp.asarray(x.shape[0]),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        sbm.prepare_sbm_compact_structure(active_mask, OTHER_INDICES),
    )
    return state, arguments


def test_compact_sbm_packed_api_is_public() -> None:
    assert kernels.SBMPackedChainResult is sbm.SBMPackedChainResult
    assert (
        kernels.sample_compact_sbm_packed_chain is sbm.sample_compact_sbm_packed_chain
    )
    assert (
        kernels.sample_compact_sbm_packed_chains is sbm.sample_compact_sbm_packed_chains
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_compact_sbm_packed_chain_exactly_reconstructs_full_chain(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    state, arguments = _compact_chain_case(dtype)
    key = jax.random.key(311)

    packed = jax.jit(
        sbm.sample_compact_sbm_packed_chain,
        static_argnames=("burnin", "n_samples"),
    )(key, state, *arguments, burnin=2, n_samples=3)
    full = sbm.sample_compact_sbm_chain(
        key,
        state,
        *arguments,
        burnin=2,
        n_samples=3,
    )

    assert packed.covariance.shape == (3, 6)
    assert packed.phi.shape == (3, 6)
    assert packed.accepted.shape == (5,)
    assert packed.covariance.dtype == dtype
    assert packed.phi.dtype == dtype
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(packed.covariance, dimension=3),
        full.covariance,
    )
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(packed.phi, dimension=3),
        full.phi,
    )
    assert jnp.array_equal(packed.accepted, full.accepted)
    for actual, expected in zip(packed.final_state, full.final_state, strict=True):
        assert jnp.array_equal(actual, expected)


def test_compact_sbm_packed_chains_match_independent_full_chains() -> None:
    state, arguments = _compact_chain_case(jnp.float32)
    keys = jax.random.split(jax.random.key(313), 2)
    states = jax.tree.map(lambda value: jnp.stack([value, value]), state)

    packed = jax.jit(
        sbm.sample_compact_sbm_packed_chains,
        static_argnames=("burnin", "n_samples"),
    )(keys, states, *arguments, burnin=1, n_samples=2)
    full = [
        sbm.sample_compact_sbm_chain(
            key,
            state,
            *arguments,
            burnin=1,
            n_samples=2,
        )
        for key in keys
    ]

    assert packed.covariance.shape == (2, 2, 6)
    assert packed.phi.shape == (2, 2, 6)
    assert packed.accepted.shape == (2, 3)
    assert jnp.allclose(
        unpack_lower_triangle_column_major(packed.covariance, dimension=3),
        jnp.stack([chain.covariance for chain in full]),
    )
    assert jnp.allclose(
        unpack_lower_triangle_column_major(packed.phi, dimension=3),
        jnp.stack([chain.phi for chain in full]),
    )
    assert jnp.array_equal(
        packed.accepted,
        jnp.stack([chain.accepted for chain in full]),
    )
    for index, actual in enumerate(packed.final_state):
        assert jnp.allclose(
            actual,
            jnp.stack([chain.final_state[index] for chain in full]),
        )


def test_compact_sbm_packed_chain_supports_zero_width_structure() -> None:
    active_mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    state, arguments = _compact_chain_case(
        jnp.float32,
        active_mask=active_mask,
    )
    key = jax.random.key(317)

    packed = sbm.sample_compact_sbm_packed_chain(
        key,
        state,
        *arguments,
        burnin=1,
        n_samples=2,
    )
    full = sbm.sample_compact_sbm_chain(
        key,
        state,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    assert arguments[-1].active_positions.shape == (3, 0)
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(packed.covariance, dimension=3),
        full.covariance,
    )
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(packed.phi, dimension=3),
        full.phi,
    )
    assert jnp.array_equal(packed.accepted, full.accepted)


def test_compact_sbm_packed_chain_retains_r_gamma_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = jnp.float32
    active_mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    state, arguments = _compact_chain_case(dtype, active_mask=active_mask)
    proposed_gamma = jnp.asarray(5e-7, dtype=dtype)

    def accepted_gamma(*_: object) -> GIGSample:
        return GIGSample(
            value=proposed_gamma,
            accepted=jnp.asarray(True),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(sbm, "sample_gig", accepted_gamma)
    key = jax.random.key(318)

    packed = sbm.sample_compact_sbm_packed_chain(
        key,
        state,
        *arguments,
        burnin=1,
        n_samples=2,
    )
    full = sbm.sample_compact_sbm_chain(
        key,
        state,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    reconstructed = unpack_lower_triangle_column_major(
        packed.covariance,
        dimension=3,
    )
    expected_covariance = jnp.broadcast_to(
        jnp.eye(3, dtype=dtype) * jnp.asarray(1e-6, dtype=dtype),
        (2, 3, 3),
    )
    assert jnp.array_equal(reconstructed, expected_covariance)
    assert jnp.array_equal(reconstructed, full.covariance)
    assert jnp.array_equal(packed.accepted, full.accepted)


def test_compact_sbm_packed_chain_preserves_state_on_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _compact_chain_case(jnp.float32)

    def reject_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del key, lambda_, psi
        return GIGSample(
            value=jnp.full_like(chi, 5e-7),
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

    monkeypatch.setattr(sbm, "sample_gig", reject_gamma)
    monkeypatch.setattr(sbm, "_sample_gig_batch", accept_phi)

    result = sbm.sample_compact_sbm_packed_chain(
        jax.random.key(319),
        state,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    assert jnp.array_equal(result.accepted, jnp.zeros((3,), dtype=jnp.bool_))
    expected_covariance = jnp.broadcast_to(
        state.covariance,
        (2, *state.covariance.shape),
    )
    expected_phi = jnp.broadcast_to(state.phi, (2, *state.phi.shape))
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(result.covariance, dimension=3),
        expected_covariance,
    )
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(result.phi, dimension=3),
        expected_phi,
    )
    for actual, expected in zip(result.final_state, state, strict=True):
        assert jnp.array_equal(actual, expected)


def test_compact_sbm_packed_chains_validate_keys_and_state_axis() -> None:
    state, arguments = _compact_chain_case(jnp.float32)
    states = jax.tree.map(lambda value: jnp.stack([value, value]), state)

    with pytest.raises(TypeError, match="typed JAX keys"):
        sbm.sample_compact_sbm_packed_chains(
            jax.random.PRNGKey(331),
            states,
            *arguments,
            burnin=0,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="one-dimensional batch"):
        sbm.sample_compact_sbm_packed_chains(
            jax.random.split(jax.random.key(337), 2)[:, None],
            states,
            *arguments,
            burnin=0,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="at least one chain"):
        sbm.sample_compact_sbm_packed_chains(
            jax.random.split(jax.random.key(347), 0),
            jax.tree.map(lambda value: value[:0], states),
            *arguments,
            burnin=0,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="state leading dimension"):
        sbm.sample_compact_sbm_packed_chains(
            jax.random.split(jax.random.key(349), 3),
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
def test_compact_sbm_packed_chain_validates_static_lengths(
    burnin: int,
    n_samples: int,
    message: str,
) -> None:
    state, arguments = _compact_chain_case(jnp.float32)

    with pytest.raises(ValueError, match=message):
        sbm.sample_compact_sbm_packed_chain(
            jax.random.key(353),
            state,
            *arguments,
            burnin=burnin,
            n_samples=n_samples,
        )


def test_compact_sbm_packed_chain_does_not_materialize_burnin_draws() -> None:
    state, arguments = _compact_chain_case(jnp.float32)
    traced = jax.make_jaxpr(
        lambda key, initial, *args: sbm.sample_compact_sbm_packed_chain(
            key,
            initial,
            *args,
            burnin=2,
            n_samples=3,
        )
    )(jax.random.key(359), state, *arguments)
    scans = [
        equation for equation in traced.jaxpr.eqns if equation.primitive.name == "scan"
    ]

    assert len(scans) == 2
    num_carry = len(jax.tree.leaves(state))
    burnin_outputs = scans[0].outvars[num_carry:]
    burnin_shapes = [
        getattr(variable.aval, "shape", None) for variable in scans[0].outvars
    ]
    posterior_shapes = [
        getattr(variable.aval, "shape", None) for variable in scans[1].outvars
    ]
    assert len(burnin_outputs) == 1
    assert burnin_outputs[0].aval.shape == (2,)
    assert burnin_outputs[0].aval.dtype == jnp.bool_
    assert (2, 6) not in burnin_shapes
    assert (2, 3, 3) not in burnin_shapes
    assert (3, 6) in posterior_shapes
    assert (3, 3, 3) not in posterior_shapes
