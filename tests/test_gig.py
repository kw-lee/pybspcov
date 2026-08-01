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
