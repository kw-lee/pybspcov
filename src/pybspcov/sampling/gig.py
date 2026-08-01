"""Bounded generalized inverse Gaussian sampling in pure JAX."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class GIGSample(NamedTuple):
    """A GIG draw and bounded rejection-sampler status."""

    value: Array
    accepted: Array
    iterations: Array


def _standardized_mode(shape: Array, omega: Array) -> Array:
    above_one = (
        jnp.sqrt(jnp.square(shape - 1.0) + jnp.square(omega)) + shape - 1.0
    ) / omega
    below_one = omega / (
        jnp.sqrt(jnp.square(1.0 - shape) + jnp.square(omega)) + 1.0 - shape
    )
    return jnp.where(shape >= 1.0, above_one, below_one)


def sample_gig(
    key: Array,
    lambda_: Array,
    chi: Array,
    psi: Array,
    *,
    max_iterations: int = 256,
) -> GIGSample:
    """Draw one ``GIG(lambda_, chi, psi)`` variate.

    This first kernel implements the no-shift ratio-of-uniforms regime used by
    GIGrvg. Unsupported small-omega cases and exhausted rejection loops return
    ``accepted=False`` so the compiled caller can handle them explicitly.
    """
    shape = jnp.abs(lambda_)
    omega = jnp.sqrt(psi * chi)
    alpha = jnp.sqrt(chi / psi)
    supported = (
        jnp.isfinite(lambda_)
        & jnp.isfinite(chi)
        & jnp.isfinite(psi)
        & (chi > 0.0)
        & (psi > 0.0)
        & ((shape >= 1.0 - 2.25 * jnp.square(omega)) | (omega > 0.2))
    )

    t = 0.5 * (shape - 1.0)
    s = 0.25 * omega
    mode = _standardized_mode(shape, omega)
    log_normalizer = t * jnp.log(mode) - s * (mode + 1.0 / mode)
    upper_mode = (
        shape + 1.0 + jnp.sqrt(jnp.square(shape + 1.0) + jnp.square(omega))
    ) / omega
    upper_u = jnp.exp(
        0.5 * (shape + 1.0) * jnp.log(upper_mode)
        - s * (upper_mode + 1.0 / upper_mode)
        - log_normalizer
    )

    dtype = jnp.result_type(lambda_, chi, psi)
    initial = (
        key,
        jnp.asarray(jnp.nan, dtype=dtype),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def condition(state: tuple[Array, Array, Array, Array]) -> Array:
        _, _, accepted, iterations = state
        return (~accepted) & (iterations < max_iterations)

    def draw(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        current_key, value, accepted, iterations = state
        current_key, u_key, v_key = jax.random.split(current_key, 3)
        u = upper_u * jax.random.uniform(u_key, dtype=dtype)
        v = jax.random.uniform(v_key, dtype=dtype)
        v = jnp.maximum(v, jnp.asarray(1e-30, dtype=dtype))
        standardized = u / v
        log_density = (
            t * jnp.log(standardized)
            - s * (standardized + 1.0 / standardized)
            - log_normalizer
        )
        accepted = (
            (standardized > 0.0)
            & jnp.isfinite(standardized)
            & (jnp.log(v) <= log_density)
        )
        candidate = jnp.where(lambda_ < 0.0, alpha / standardized, alpha * standardized)
        value = jnp.where(accepted, candidate, value)
        return current_key, value, accepted, iterations + 1

    _, value, accepted, iterations = jax.lax.while_loop(condition, draw, initial)
    accepted = accepted & supported
    value = jnp.where(accepted, value, jnp.asarray(jnp.nan, dtype=dtype))
    return GIGSample(value=value, accepted=accepted, iterations=iterations)
