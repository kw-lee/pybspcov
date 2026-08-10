"""Bounded generalized inverse Gaussian sampling in pure JAX."""

from collections.abc import Callable
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


_ScalarState = tuple[Array, Array, Array, Array]


def _run_scalar_rejection_loop(
    condition: Callable[[_ScalarState], Array],
    draw: Callable[[_ScalarState], _ScalarState],
    initial: _ScalarState,
    proposal_chunk_size: int,
) -> _ScalarState:
    """Run scalar proposals in exact, acceptance-preserving chunks."""
    if proposal_chunk_size == 1:
        return jax.lax.while_loop(condition, draw, initial)

    def draw_one(
        state: _ScalarState,
        _: None,
    ) -> tuple[_ScalarState, None]:
        active = condition(state)
        proposed = draw(state)
        selected = jax.tree.map(
            lambda proposed_value, current_value: jax.lax.select(
                active,
                proposed_value,
                current_value,
            ),
            proposed,
            state,
        )
        return selected, None

    def draw_chunk(state: _ScalarState) -> _ScalarState:
        state, _ = jax.lax.scan(
            draw_one,
            state,
            xs=None,
            length=proposal_chunk_size,
        )
        return state

    return jax.lax.while_loop(condition, draw_chunk, initial)


def _sample_small_omega(
    key: Array,
    lambda_: Array,
    shape: Array,
    omega: Array,
    alpha: Array,
    max_iterations: int,
    proposal_chunk_size: int,
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

    _, value, accepted, iterations = _run_scalar_rejection_loop(
        condition, draw, initial, proposal_chunk_size
    )
    return GIGSample(value=value, accepted=accepted, iterations=iterations)


def _sample_no_shift(
    key: Array,
    lambda_: Array,
    shape: Array,
    omega: Array,
    alpha: Array,
    max_iterations: int,
    proposal_chunk_size: int,
) -> GIGSample:
    """Sample the no-shift ratio-of-uniforms GIGrvg regime."""
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

    dtype = jnp.result_type(lambda_, omega, alpha)
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

    _, value, accepted, iterations = _run_scalar_rejection_loop(
        condition, draw, initial, proposal_chunk_size
    )
    return GIGSample(value=value, accepted=accepted, iterations=iterations)


def _empty_batch_sample(batch_size: int, dtype: jnp.dtype) -> GIGSample:
    return GIGSample(
        value=jnp.full((batch_size,), jnp.nan, dtype=dtype),
        accepted=jnp.zeros((batch_size,), dtype=jnp.bool_),
        iterations=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def _run_masked_batch(
    keys: Array,
    regime_mask: Array,
    dtype: jnp.dtype,
    max_iterations: int,
    proposal_chunk_size: int,
    propose: Callable[[Array], tuple[Array, Array, Array]],
) -> GIGSample:
    """Run one rejection loop for all lanes in a GIG regime."""
    initial_sample = _empty_batch_sample(regime_mask.shape[0], dtype)
    initial = (
        keys,
        initial_sample.value,
        initial_sample.accepted,
        initial_sample.iterations,
    )

    def condition(state: tuple[Array, Array, Array, Array]) -> Array:
        _, _, accepted, iterations = state
        active = regime_mask & (~accepted) & (iterations < max_iterations)
        return jnp.any(active)

    def draw_one(
        state: tuple[Array, Array, Array, Array], _: None
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
        current_keys, values, accepted, iterations = state
        active = regime_mask & (~accepted) & (iterations < max_iterations)
        next_keys, candidates, proposal_accepted = propose(current_keys)
        accepted_now = active & proposal_accepted
        values = jnp.where(accepted_now, candidates, values)
        accepted = accepted | accepted_now
        iterations = iterations + active.astype(jnp.int32)
        return (next_keys, values, accepted, iterations), None

    def draw_chunk(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        state, _ = jax.lax.scan(
            draw_one,
            state,
            xs=None,
            length=proposal_chunk_size,
        )
        return state

    _, values, accepted, iterations = jax.lax.while_loop(condition, draw_chunk, initial)
    return GIGSample(value=values, accepted=accepted, iterations=iterations)


def _sample_gig_batch(
    keys: Array,
    lambda_: Array,
    chi: Array,
    psi: Array,
    *,
    max_iterations: int = 256,
    proposal_chunk_size: int = 1,
) -> GIGSample:
    """Draw a one-dimensional batch of GIG variates with masked loops."""
    if not isinstance(proposal_chunk_size, int) or proposal_chunk_size < 1:
        raise ValueError("proposal_chunk_size must be a positive integer")

    batch_size = keys.shape[0]
    lambda_ = jnp.broadcast_to(lambda_, (batch_size,))
    chi = jnp.broadcast_to(chi, (batch_size,))
    psi = jnp.broadcast_to(psi, (batch_size,))
    shape = jnp.abs(lambda_)
    omega = jnp.sqrt(psi * chi)
    alpha = jnp.sqrt(chi / psi)
    dtype = jnp.result_type(lambda_, omega, alpha)
    valid = (
        jnp.isfinite(lambda_)
        & jnp.isfinite(chi)
        & jnp.isfinite(psi)
        & (chi > 0.0)
        & (psi > 0.0)
    )
    no_shift_supported = (shape >= 1.0 - 2.25 * jnp.square(omega)) | (omega > 0.2)
    small_mask = valid & (~no_shift_supported)
    no_shift_mask = valid & no_shift_supported

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

    def split_keys(current_keys: Array) -> tuple[Array, Array, Array]:
        split_batch = jax.vmap(lambda key: jax.random.split(key, 3))(current_keys)
        return split_batch[:, 0], split_batch[:, 1], split_batch[:, 2]

    def uniforms(uniform_keys: Array) -> Array:
        return jax.vmap(lambda key: jax.random.uniform(key, dtype=dtype))(uniform_keys)

    def propose_small(current_keys: Array) -> tuple[Array, Array, Array]:
        next_keys, region_keys, accept_keys = split_keys(current_keys)
        region = total_area * uniforms(region_keys)
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
        uniform = uniforms(accept_keys) * hat
        log_density = (shape - 1.0) * jnp.log(standardized) - 0.5 * omega * (
            standardized + 1.0 / standardized
        )
        accepted = (
            (standardized > 0.0)
            & jnp.isfinite(standardized)
            & (jnp.log(uniform) <= log_density)
        )
        candidate = jnp.where(lambda_ < 0.0, alpha / standardized, alpha * standardized)
        return next_keys, candidate, accepted

    t = 0.5 * (shape - 1.0)
    s = 0.25 * omega
    log_normalizer = t * jnp.log(mode) - s * (mode + 1.0 / mode)
    upper_mode = (
        shape + 1.0 + jnp.sqrt(jnp.square(shape + 1.0) + jnp.square(omega))
    ) / omega
    upper_u = jnp.exp(
        0.5 * (shape + 1.0) * jnp.log(upper_mode)
        - s * (upper_mode + 1.0 / upper_mode)
        - log_normalizer
    )

    def propose_no_shift(current_keys: Array) -> tuple[Array, Array, Array]:
        next_keys, u_keys, v_keys = split_keys(current_keys)
        u = upper_u * uniforms(u_keys)
        v = jnp.maximum(uniforms(v_keys), jnp.asarray(1e-30, dtype=dtype))
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
        return next_keys, candidate, accepted

    def empty(_: None) -> GIGSample:
        return _empty_batch_sample(batch_size, dtype)

    small_draws = jax.lax.cond(
        jnp.any(small_mask),
        lambda _: _run_masked_batch(
            keys,
            small_mask,
            dtype,
            max_iterations,
            proposal_chunk_size,
            propose_small,
        ),
        empty,
        operand=None,
    )
    no_shift_draws = jax.lax.cond(
        jnp.any(no_shift_mask),
        lambda _: _run_masked_batch(
            keys,
            no_shift_mask,
            dtype,
            max_iterations,
            proposal_chunk_size,
            propose_no_shift,
        ),
        empty,
        operand=None,
    )
    accepted = (
        jnp.where(small_mask, small_draws.accepted, no_shift_draws.accepted) & valid
    )
    iterations = jnp.where(
        small_mask, small_draws.iterations, no_shift_draws.iterations
    )
    value = jnp.where(small_mask, small_draws.value, no_shift_draws.value)
    value = jnp.where(accepted, value, jnp.asarray(jnp.nan, dtype=dtype))
    return GIGSample(value=value, accepted=accepted, iterations=iterations)


def sample_gig(
    key: Array,
    lambda_: Array,
    chi: Array,
    psi: Array,
    *,
    max_iterations: int = 256,
    proposal_chunk_size: int = 8,
) -> GIGSample:
    """Draw one ``GIG(lambda_, chi, psi)`` variate.

    The implementation follows the no-shift ratio-of-uniforms and small-omega
    rejection regimes used by GIGrvg. Exhausted rejection loops return
    ``accepted=False`` so the compiled caller can handle them explicitly.
    """
    if not isinstance(proposal_chunk_size, int) or proposal_chunk_size < 1:
        raise ValueError("proposal_chunk_size must be a positive integer")

    shape = jnp.abs(lambda_)
    omega = jnp.sqrt(psi * chi)
    alpha = jnp.sqrt(chi / psi)
    dtype = jnp.result_type(lambda_, omega, alpha)
    valid = (
        jnp.isfinite(lambda_)
        & jnp.isfinite(chi)
        & jnp.isfinite(psi)
        & (chi > 0.0)
        & (psi > 0.0)
    )
    no_shift_supported = (shape >= 1.0 - 2.25 * jnp.square(omega)) | (omega > 0.2)

    use_small = valid & (~no_shift_supported)
    operands = (key, lambda_, shape, omega, alpha)

    def sample_small(values: tuple[Array, Array, Array, Array, Array]) -> GIGSample:
        branch_key, branch_lambda, branch_shape, branch_omega, branch_alpha = values
        return _sample_small_omega(
            branch_key,
            branch_lambda,
            branch_shape,
            branch_omega,
            branch_alpha,
            max_iterations,
            proposal_chunk_size,
        )

    def sample_no_shift(values: tuple[Array, Array, Array, Array, Array]) -> GIGSample:
        branch_key, branch_lambda, branch_shape, branch_omega, branch_alpha = values
        return _sample_no_shift(
            branch_key,
            branch_lambda,
            branch_shape,
            branch_omega,
            branch_alpha,
            max_iterations,
            proposal_chunk_size,
        )

    sample = jax.lax.cond(
        use_small,
        sample_small,
        sample_no_shift,
        operands,
    )
    accepted = sample.accepted & valid
    value = jnp.where(
        accepted,
        sample.value,
        jnp.asarray(jnp.nan, dtype=dtype),
    )
    return GIGSample(value=value, accepted=accepted, iterations=sample.iterations)
