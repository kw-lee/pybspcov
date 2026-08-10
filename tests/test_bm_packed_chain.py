import jax
import jax.numpy as jnp
import pytest

import pybspcov.kernels.bm as bm_kernel
from pybspcov.kernels import (
    pack_lower_triangle_column_major,
    sample_bm_packed_chain,
    sample_bm_packed_chains,
    unpack_lower_triangle_column_major,
)
from pybspcov.kernels.bm import (
    BMState,
    initialize_bm_state,
    sample_bm_chain,
    sample_bm_chains,
)
from pybspcov.sampling.gig import GIGSample


def _bm_chain_case(
    dtype: jnp.dtype,
) -> tuple[BMState, tuple[jax.Array, ...]]:
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


def test_lower_triangle_helpers_use_r_column_major_order() -> None:
    matrix = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    matrices = jnp.stack([matrix, 2.0 * matrix])

    packed = jax.jit(pack_lower_triangle_column_major)(matrices)
    reconstructed = jax.jit(
        lambda values: unpack_lower_triangle_column_major(values, dimension=3)
    )(packed)

    assert jnp.array_equal(packed[0], jnp.arange(1.0, 7.0))
    assert jnp.array_equal(reconstructed, matrices)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        (jnp.ones((3,)), "square dimensions"),
        (jnp.ones((2, 3)), "square dimensions"),
    ],
)
def test_pack_lower_triangle_rejects_non_square_inputs(
    matrix: jax.Array,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        pack_lower_triangle_column_major(matrix)


def test_unpack_lower_triangle_validates_dimension_and_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        unpack_lower_triangle_column_major(jnp.ones((1,)), dimension=0)
    with pytest.raises(ValueError, match="trailing lower-triangle"):
        unpack_lower_triangle_column_major(jnp.asarray(1.0), dimension=1)
    with pytest.raises(ValueError, match="must be 6"):
        unpack_lower_triangle_column_major(jnp.ones((5,)), dimension=3)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sample_bm_packed_chain_reconstructs_full_chain_in_r_order(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    state, arguments = _bm_chain_case(dtype)
    key = jax.random.key(101)
    run_packed = jax.jit(
        sample_bm_packed_chain,
        static_argnames=("burnin", "n_samples"),
    )

    packed = run_packed(
        key,
        state,
        *arguments,
        burnin=1,
        n_samples=3,
    )
    full = sample_bm_chain(
        key,
        state,
        *arguments,
        burnin=1,
        n_samples=3,
    )

    assert packed.covariance.shape == (3, 6)
    assert packed.phi.shape == (3, 6)
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


def test_sample_bm_packed_chains_matches_full_multichain_output() -> None:
    state, arguments = _bm_chain_case(jnp.float32)
    keys = jax.random.split(jax.random.key(202), 2)
    states = BMState(*(jnp.stack([value, value]) for value in state))

    packed = sample_bm_packed_chains(
        keys,
        states,
        *arguments,
        burnin=1,
        n_samples=2,
    )
    full = sample_bm_chains(
        keys,
        states,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    assert packed.covariance.shape == (2, 2, 6)
    assert packed.phi.shape == (2, 2, 6)
    assert packed.accepted.shape == (2, 3)
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(packed.covariance, dimension=3),
        full.covariance,
    )
    assert jnp.array_equal(
        unpack_lower_triangle_column_major(packed.phi, dimension=3),
        full.phi,
    )
    assert jnp.array_equal(packed.accepted, full.accepted)


def test_sample_bm_packed_chain_retains_rolled_back_rejected_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _bm_chain_case(jnp.float32)

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
    result = jax.jit(
        bm_kernel.sample_bm_packed_chain,
        static_argnames=("burnin", "n_samples"),
    )(
        jax.random.key(211),
        state,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    expected_covariance = pack_lower_triangle_column_major(state.covariance)
    expected_phi = pack_lower_triangle_column_major(state.phi)
    assert jnp.array_equal(result.accepted, jnp.asarray([False, False, False]))
    assert jnp.array_equal(
        result.covariance,
        jnp.broadcast_to(expected_covariance, result.covariance.shape),
    )
    assert jnp.array_equal(
        result.phi,
        jnp.broadcast_to(expected_phi, result.phi.shape),
    )
    for actual, expected in zip(result.final_state, state, strict=True):
        assert jnp.array_equal(actual, expected)


def test_sample_bm_packed_chain_validates_static_lengths() -> None:
    state, arguments = _bm_chain_case(jnp.float32)
    key = jax.random.key(303)

    with pytest.raises(ValueError, match="burnin"):
        sample_bm_packed_chain(key, state, *arguments, burnin=-1, n_samples=1)
    with pytest.raises(ValueError, match="n_samples"):
        sample_bm_packed_chain(key, state, *arguments, burnin=0, n_samples=0)


def test_sample_bm_packed_chains_rejects_legacy_keys() -> None:
    state, arguments = _bm_chain_case(jnp.float32)
    states = BMState(*(value[jnp.newaxis, ...] for value in state))

    with pytest.raises(TypeError, match="typed JAX keys"):
        sample_bm_packed_chains(
            jnp.asarray([1, 2], dtype=jnp.uint32),
            states,
            *arguments,
            burnin=0,
            n_samples=1,
        )


def test_sample_bm_packed_chain_does_not_materialize_burnin_draws() -> None:
    state, arguments = _bm_chain_case(jnp.float32)
    traced = jax.make_jaxpr(
        lambda key, initial, *args: sample_bm_packed_chain(
            key,
            initial,
            *args,
            burnin=2,
            n_samples=3,
        )
    )(jax.random.key(307), state, *arguments)
    scans = [
        equation for equation in traced.jaxpr.eqns if equation.primitive.name == "scan"
    ]

    assert len(scans) == 2
    burnin_shapes = [
        getattr(variable.aval, "shape", None) for variable in scans[0].outvars
    ]
    posterior_shapes = [
        getattr(variable.aval, "shape", None) for variable in scans[1].outvars
    ]
    assert (2, 6) not in burnin_shapes
    assert (3, 6) in posterior_shapes
