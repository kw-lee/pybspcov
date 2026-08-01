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


def _sample_small_omega(
    key: Array,
    lambda_: Array,
    shape: Array,
    omega: Array,
    alpha: Array,
    max_iterations: int,
) -> GIGSample:
    """Sample the small-omega, shape-below-one GIGrvg regime."""
    dtype = jnp.result_type(lambda_, omega, alpha)
    mode = _standardized_mode(shape, omega)
    split = omega / (1.0 - shape)
    tail_start = jnp.maximum(split, 2.0 / omega)
    k0 = jnp.exp((shape - 1.0) * jnp.log(mode) - 0.5 * omega * (mode + 1.0 / mode))
    area0 = k0 * split
    k1 = jnp.exp(-omega)
    safe_shape = jnp.maximum(shape, jnp.asarray(1e-12, dtype=dtype))
    area1_power = (
        k1 / safe_shape * (jnp.power(2.0 / omega, shape) - jnp.power(split, shape))
    )
    area1_log = k1 * jnp.log(2.0 / jnp.square(omega))
    area1 = jnp.where(
        split >= 2.0 / omega,
        0.0,
        jnp.where(shape == 0.0, area1_log, area1_power),
    )
    k2 = jnp.power(tail_start, shape - 1.0)
    area2 = k2 * 2.0 * jnp.exp(-0.5 * omega * tail_start) / omega
    total_area = area0 + area1 + area2
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
        current_key, region_key, accept_key = jax.random.split(current_key, 3)
        region = total_area * jax.random.uniform(region_key, dtype=dtype)
        first_x = split * region / area0
        middle_area = region - area0
        middle_x_log = omega * jnp.exp(jnp.exp(omega) * middle_area)
        middle_x_power = jnp.power(
            jnp.power(split, shape) + shape / k1 * middle_area,
            1.0 / safe_shape,
        )
        middle_x = jnp.where(shape == 0.0, middle_x_log, middle_x_power)
        tail_area = middle_area - area1
        tail_x = (
            -2.0
            / omega
            * jnp.log(
                jnp.exp(-0.5 * omega * tail_start) - omega / (2.0 * k2) * tail_area
            )
        )
        in_first = region <= area0
        in_middle = (~in_first) & (middle_area <= area1)
        standardized = jnp.where(
            in_first, first_x, jnp.where(in_middle, middle_x, tail_x)
        )
        hat = jnp.where(
            in_first,
            k0,
            jnp.where(
                in_middle,
                k1 * jnp.power(standardized, shape - 1.0),
                k2 * jnp.exp(-0.5 * omega * standardized),
            ),
        )
        uniform = jax.random.uniform(accept_key, dtype=dtype) * hat
        log_density = (shape - 1.0) * jnp.log(standardized) - 0.5 * omega * (
            standardized + 1.0 / standardized
        )
        accepted = (
            (standardized > 0.0)
            & jnp.isfinite(standardized)
            & (jnp.log(uniform) <= log_density)
        )
        candidate = jnp.where(lambda_ < 0.0, alpha / standardized, alpha * standardized)
        value = jnp.where(accepted, candidate, value)
        return current_key, value, accepted, iterations + 1

    _, value, accepted, iterations = jax.lax.while_loop(condition, draw, initial)
    return GIGSample(value=value, accepted=accepted, iterations=iterations)


def sample_gig(
    key: Array,
    lambda_: Array,
    chi: Array,
    psi: Array,
    *,
    max_iterations: int = 256,
) -> GIGSample:
    """Draw one ``GIG(lambda_, chi, psi)`` variate.

    The implementation follows the no-shift ratio-of-uniforms and small-omega
    rejection regimes used by GIGrvg. Exhausted rejection loops return
    ``accepted=False`` so the compiled caller can handle them explicitly.
    """
    shape = jnp.abs(lambda_)
    omega = jnp.sqrt(psi * chi)
    alpha = jnp.sqrt(chi / psi)
    valid = (
        jnp.isfinite(lambda_)
        & jnp.isfinite(chi)
        & jnp.isfinite(psi)
        & (chi > 0.0)
        & (psi > 0.0)
    )
    no_shift_supported = (shape >= 1.0 - 2.25 * jnp.square(omega)) | (omega > 0.2)

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
    small_sample = _sample_small_omega(
        key, lambda_, shape, omega, alpha, max_iterations
    )
    use_small = valid & (~no_shift_supported)
    accepted = jnp.where(use_small, small_sample.accepted, accepted & valid)
    value = jnp.where(use_small, small_sample.value, value)
    iterations = jnp.where(use_small, small_sample.iterations, iterations)
    value = jnp.where(accepted, value, jnp.asarray(jnp.nan, dtype=dtype))
    return GIGSample(value=value, accepted=accepted, iterations=iterations)
