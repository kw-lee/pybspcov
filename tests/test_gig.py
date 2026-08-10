import jax
import jax.numpy as jnp
import pytest
from scipy.special import kv

from pybspcov.sampling.gig import _sample_gig_batch, sample_gig


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_gig_batch_matches_scalar_draws_for_heterogeneous_parameters(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    keys = jax.random.split(jax.random.key(29), 4)
    lambda_ = jnp.asarray([-2.0, 0.0, 1.5, -0.25], dtype=dtype)
    chi = jnp.asarray([2.0, 0.01, 0.5, 0.04], dtype=dtype)
    psi = jnp.asarray([1.0, 1.0, 3.0, 0.5], dtype=dtype)

    batched = jax.jit(_sample_gig_batch)(keys, lambda_, chi, psi)
    scalar = jax.vmap(sample_gig)(keys, lambda_, chi, psi)

    assert batched.value.dtype == dtype
    assert jnp.array_equal(batched.accepted, scalar.accepted)
    assert jnp.array_equal(batched.iterations, scalar.iterations)
    assert jnp.allclose(batched.value, scalar.value, rtol=0.0, atol=0.0)


def test_gig_batch_size_one_chunk_one_matches_scalar_exactly() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    key = jax.random.key(31)
    keys = jax.random.split(key, 1)
    lambda_ = jnp.asarray([-2.0], dtype=jnp.float64)
    chi = jnp.asarray([2.0], dtype=jnp.float64)
    psi = jnp.asarray([1.0], dtype=jnp.float64)

    batched = _sample_gig_batch(
        keys,
        lambda_,
        chi,
        psi,
        proposal_chunk_size=1,
    )
    scalar = sample_gig(keys[0], lambda_[0], chi[0], psi[0])

    assert batched.value[0] == scalar.value
    assert batched.accepted[0] == scalar.accepted
    assert batched.iterations[0] == scalar.iterations


@pytest.mark.parametrize("proposal_chunk_size", [2, 4])
def test_gig_batch_chunked_proposals_match_single_proposal_loop(
    proposal_chunk_size: int,
) -> None:
    keys = jax.random.split(jax.random.key(33), 4)
    lambda_ = jnp.asarray([-2.0, 0.0, 1.5, -0.25], dtype=jnp.float32)
    chi = jnp.asarray([2.0, 0.01, 0.5, 0.04], dtype=jnp.float32)
    psi = jnp.asarray([1.0, 1.0, 3.0, 0.5], dtype=jnp.float32)

    single = _sample_gig_batch(keys, lambda_, chi, psi, proposal_chunk_size=1)
    chunked = _sample_gig_batch(
        keys,
        lambda_,
        chi,
        psi,
        proposal_chunk_size=proposal_chunk_size,
    )

    assert jnp.array_equal(chunked.accepted, single.accepted)
    assert jnp.array_equal(chunked.iterations, single.iterations)
    assert jnp.allclose(chunked.value, single.value, rtol=0.0, atol=0.0)


def test_gig_batch_chunked_proposals_respect_iteration_bound() -> None:
    keys = jax.random.split(jax.random.key(35), 4)
    draws = _sample_gig_batch(
        keys,
        jnp.asarray([-2.0, 0.0, 1.5, -0.25]),
        jnp.asarray([2.0, 0.01, 0.5, 0.04]),
        jnp.asarray([1.0, 1.0, 3.0, 0.5]),
        max_iterations=1,
        proposal_chunk_size=4,
    )

    assert jnp.all(draws.iterations <= 1)


def test_gig_batch_reports_invalid_and_exhausted_lanes_independently() -> None:
    keys = jax.random.split(jax.random.key(37), 4)
    draws = _sample_gig_batch(
        keys,
        jnp.asarray([0.0, 0.0, -2.0, 0.0]),
        jnp.asarray([0.0, 0.01, 2.0, -1.0]),
        jnp.ones(4),
        max_iterations=0,
    )

    assert jnp.array_equal(draws.accepted, jnp.zeros(4, dtype=jnp.bool_))
    assert jnp.all(jnp.isnan(draws.value))
    assert jnp.array_equal(draws.iterations, jnp.zeros(4, dtype=jnp.int32))


def test_gig_batch_keeps_invalid_lane_isolated_from_valid_lanes() -> None:
    keys = jax.random.split(jax.random.key(41), 3)
    draws = jax.jit(_sample_gig_batch)(
        keys,
        jnp.asarray([-2.0, 0.0, 0.0]),
        jnp.asarray([2.0, 0.0, 0.01]),
        jnp.ones(3),
    )

    assert jnp.array_equal(draws.accepted, jnp.asarray([True, False, True]))
    assert jnp.isfinite(draws.value[0])
    assert jnp.isfinite(draws.value[2])
    assert jnp.isnan(draws.value[1])
    assert draws.iterations[1] == 0


def test_gig_batch_has_one_masked_loop_per_regime() -> None:
    keys = jax.random.split(jax.random.key(43), 4)
    jaxpr = jax.make_jaxpr(_sample_gig_batch)(
        keys,
        jnp.asarray([-2.0, 0.0, 1.5, -0.25]),
        jnp.asarray([2.0, 0.01, 0.5, 0.04]),
        jnp.asarray([1.0, 1.0, 3.0, 0.5]),
    )

    cond_equations = [
        equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "cond"
    ]
    assert len(cond_equations) == 2
    for equation in cond_equations:
        branch_while_counts = [
            sum(
                branch_equation.primitive.name == "while"
                for branch_equation in branch.jaxpr.eqns
            )
            for branch in equation.params["branches"]
        ]
        assert sorted(branch_while_counts) == [0, 1]


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize(
    ("lambda_value", "chi", "psi", "expected_mean", "rtol"),
    [
        pytest.param(
            -2.0,
            2.0,
            1.0,
            jnp.sqrt(2.0) * kv(-1.0, jnp.sqrt(2.0)) / kv(-2.0, jnp.sqrt(2.0)),
            0.05,
            id="no-shift",
        ),
        pytest.param(
            0.0,
            0.01,
            1.0,
            0.1 * kv(1.0, 0.1) / kv(0.0, 0.1),
            0.08,
            id="small-omega",
        ),
    ],
)
def test_gig_batch_matches_theoretical_mean(
    dtype_name: str,
    lambda_value: float,
    chi: float,
    psi: float,
    expected_mean: float,
    rtol: float,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    batch_size = 8_192
    draws = jax.jit(_sample_gig_batch)(
        jax.random.split(jax.random.key(47), batch_size),
        jnp.asarray(lambda_value, dtype=dtype),
        jnp.asarray(chi, dtype=dtype),
        jnp.asarray(psi, dtype=dtype),
    )

    assert draws.value.dtype == dtype
    assert jnp.all(draws.accepted)
    assert jnp.all(jnp.isfinite(draws.value))
    assert jnp.all(draws.value > 0.0)
    assert jnp.isclose(jnp.mean(draws.value), expected_mean, rtol=rtol)


def test_gig_batch_promotes_integer_parameters_like_scalar_sampler() -> None:
    keys = jax.random.split(jax.random.key(53), 2)
    lambda_ = jnp.asarray([-2, 1])
    chi = jnp.asarray([2, 1])
    psi = jnp.asarray([1, 3])

    batched = _sample_gig_batch(keys, lambda_, chi, psi)
    scalar = jax.vmap(sample_gig)(keys, lambda_, chi, psi)

    assert jnp.issubdtype(batched.value.dtype, jnp.floating)
    assert batched.value.dtype == scalar.value.dtype
    assert jnp.array_equal(batched.accepted, scalar.accepted)
    assert jnp.array_equal(batched.iterations, scalar.iterations)
    assert jnp.allclose(batched.value, scalar.value, rtol=0.0, atol=0.0)


def test_gig_batch_rejects_noninteger_proposal_chunk_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _sample_gig_batch(
            jax.random.split(jax.random.key(59), 1),
            jnp.asarray([0.0]),
            jnp.asarray([0.01]),
            jnp.asarray([1.0]),
            proposal_chunk_size=1.5,  # type: ignore[arg-type]
        )


def test_gig_dispatches_rejection_regimes_with_cond() -> None:
    jaxpr = jax.make_jaxpr(sample_gig)(
        jax.random.key(3),
        jnp.asarray(0.0),
        jnp.asarray(0.01),
        jnp.asarray(1.0),
    )

    top_level_equations = jaxpr.jaxpr.eqns
    cond_equations = [
        equation
        for equation in top_level_equations
        if equation.primitive.name == "cond"
    ]
    assert len(cond_equations) == 1
    assert all(equation.primitive.name != "while" for equation in top_level_equations)

    branches = cond_equations[0].params["branches"]
    assert len(branches) == 2
    for branch in branches:
        branch_primitives = [equation.primitive.name for equation in branch.jaxpr.eqns]
        assert branch_primitives.count("while") == 1


@pytest.mark.parametrize(
    ("lambda_value", "chi", "psi"),
    [
        pytest.param(-2.0, 2.0, 1.0, id="no-shift"),
        pytest.param(0.0, 0.01, 1.0, id="small-omega"),
    ],
)
@pytest.mark.parametrize("compiled", [False, True], ids=("eager", "jit"))
def test_scalar_gig_chunked_proposals_match_single_proposal_loop(
    lambda_value: float,
    chi: float,
    psi: float,
    compiled: bool,
) -> None:
    sampler = (
        jax.jit(sample_gig, static_argnames=("proposal_chunk_size",))
        if compiled
        else sample_gig
    )
    key = jax.random.key(61)
    arguments = (
        key,
        jnp.asarray(lambda_value),
        jnp.asarray(chi),
        jnp.asarray(psi),
    )

    single = sampler(*arguments, proposal_chunk_size=1)
    chunked = sampler(*arguments, proposal_chunk_size=8)

    assert jnp.array_equal(chunked.value, single.value)
    assert jnp.array_equal(chunked.accepted, single.accepted)
    assert jnp.array_equal(chunked.iterations, single.iterations)


@pytest.mark.parametrize("proposal_chunk_size", [0, -1, 1.5])
def test_scalar_gig_rejects_invalid_proposal_chunk_size(
    proposal_chunk_size: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        sample_gig(
            jax.random.key(67),
            jnp.asarray(-2.0),
            jnp.asarray(2.0),
            jnp.asarray(1.0),
            proposal_chunk_size=proposal_chunk_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("lambda_value", "chi", "psi"),
    [
        pytest.param(-2.0, 2.0, 1.0, id="no-shift"),
        pytest.param(0.0, 0.01, 1.0, id="small-omega"),
    ],
)
def test_gig_zero_iteration_bound_exhausts_selected_regime(
    lambda_value: float, chi: float, psi: float
) -> None:
    sample = jax.jit(
        lambda key, lambda_, chi_, psi_: sample_gig(
            key,
            lambda_,
            chi_,
            psi_,
            max_iterations=0,
        )
    )(
        jax.random.key(19),
        jnp.asarray(lambda_value),
        jnp.asarray(chi),
        jnp.asarray(psi),
    )

    assert not sample.accepted
    assert jnp.isnan(sample.value)
    assert sample.iterations == 0


@pytest.mark.parametrize(
    ("lambda_value", "chi", "psi"),
    [
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, 0.0),
        (jnp.nan, 1.0, 1.0),
    ],
)
def test_gig_rejects_invalid_parameters(
    lambda_value: float, chi: float, psi: float
) -> None:
    sample = jax.jit(sample_gig)(
        jax.random.key(11),
        jnp.asarray(lambda_value),
        jnp.asarray(chi),
        jnp.asarray(psi),
    )

    assert not sample.accepted
    assert jnp.isnan(sample.value)


def test_gig_samples_are_positive_and_match_theoretical_mean() -> None:
    sample = jax.jit(sample_gig)(
        jax.random.key(7),
        jnp.array(-2.0),
        jnp.array(2.0),
        jnp.array(1.0),
    )
    assert sample.accepted
    assert jnp.isfinite(sample.value)
    assert sample.value > 0.0

    keys = jax.random.split(jax.random.key(13), 8_192)
    samples = jax.vmap(
        lambda key: (
            sample_gig(
                key,
                jnp.array(-2.0),
                jnp.array(2.0),
                jnp.array(1.0),
            ).value
        )
    )(keys)
    omega = jnp.sqrt(2.0)
    expected_mean = jnp.sqrt(2.0) * kv(-1.0, omega) / kv(-2.0, omega)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.isclose(jnp.mean(samples), expected_mean, rtol=0.04)


def test_gig_small_omega_samples_match_theoretical_mean() -> None:
    keys = jax.random.split(jax.random.key(23), 16_384)
    draws = jax.vmap(
        lambda key: sample_gig(
            key,
            jnp.array(0.0),
            jnp.array(0.01),
            jnp.array(1.0),
        )
    )(keys)

    expected_mean = 0.1 * kv(1.0, 0.1) / kv(0.0, 0.1)
    assert jnp.all(draws.accepted)
    assert jnp.all(jnp.isfinite(draws.value))
    assert jnp.isclose(jnp.mean(draws.value), expected_mean, rtol=0.06)
