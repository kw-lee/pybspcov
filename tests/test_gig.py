import jax
import jax.numpy as jnp
from scipy.special import kv

from pybspcov.sampling.gig import sample_gig


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
