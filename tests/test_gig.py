import jax
import jax.numpy as jnp
import pytest
from scipy.special import kv

from pybspcov.sampling.gig import sample_gig


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
