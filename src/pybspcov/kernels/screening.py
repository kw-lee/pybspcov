"""Fixed-shape screening masks for screened beta-mixture kernels.

The public functions validate concrete host inputs before dispatching to pure
JAX kernels. Compiled package code must validate first and then call the
explicitly unchecked private kernels.
"""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import betaln, hyp2f1


def _reject_tracer(value: Array, function_name: str) -> None:
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"{function_name} performs host validation and cannot be used inside "
            "jax.jit; call it before compilation, then use the package's validated "
            "internal kernel"
        )


def _validate_eager(predicate: Array, message: str) -> None:
    if not bool(predicate):
        raise ValueError(message)


def _validate_pairwise_shape(values: Array) -> int:
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("score_matrix must be a square two-dimensional array")
    dimension = values.shape[0]
    if dimension < 2:
        raise ValueError("score_matrix must contain at least two variables")
    return dimension


def _jeffreys_bayes_factor_from_correlation(
    correlation: Array,
    *,
    n_observations: int,
) -> Array:
    """Evaluate BayesFactor's exact ultrawide correlation Bayes factor."""
    dtype = correlation.dtype
    sample_size = jnp.asarray(n_observations, dtype=dtype)
    upper = (sample_size - 1.0) / 2.0
    lower = (sample_size + 2.0) / 2.0
    log_constant = betaln(
        (sample_size + 1.0) / 2.0,
        jnp.asarray(0.5, dtype=dtype),
    ) - jnp.log(jnp.asarray(2.0, dtype=dtype))
    perfect = jnp.abs(correlation) >= 1.0
    safe_squared_correlation = jnp.where(perfect, 0.0, jnp.square(correlation))
    log_hypergeometric = jnp.log(hyp2f1(upper, upper, lower, safe_squared_correlation))
    finite_score = jnp.exp(log_constant + log_hypergeometric)
    return jnp.where(perfect, jnp.asarray(jnp.inf, dtype=dtype), finite_score)


def _pairwise_jeffreys_bayes_factors_unchecked(x: Array) -> Array:
    """Return R-ordered lower-triangle scores without host validation."""
    centered = x - jnp.mean(x, axis=0)
    sums_of_squares = jnp.sum(jnp.square(centered), axis=0)
    scale = jnp.sqrt(jnp.outer(sums_of_squares, sums_of_squares))
    correlations = (centered.T @ centered) / scale
    dimension = x.shape[1]
    lower_mask = jnp.tril(jnp.ones((dimension, dimension), dtype=jnp.bool_), k=-1)
    safe_correlations = jnp.where(lower_mask, correlations, 0.0)
    scores = _jeffreys_bayes_factor_from_correlation(
        safe_correlations,
        n_observations=x.shape[0],
    )
    return jnp.where(lower_mask, scores, jnp.asarray(jnp.nan, dtype=x.dtype))


def pairwise_jeffreys_bayes_factors(x: Array) -> Array:
    """Compute the exact scores used by ``bspcov::pairwise.Jeffreys``.

    The calculation matches ``BayesFactor::correlationBF`` with
    ``rscale="ultrawide"``. Only the strict lower triangle is populated;
    the diagonal and upper triangle are ``NaN`` to preserve the upstream
    matrix contract.
    """
    observations = jnp.asarray(x)
    _reject_tracer(observations, "pairwise_jeffreys_bayes_factors")
    if observations.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    n_observations, dimension = observations.shape
    if n_observations < 3:
        raise ValueError("x must contain at least three observations")
    if dimension < 2:
        raise ValueError("x must contain at least two variables")
    if not jnp.issubdtype(observations.dtype, jnp.number) or jnp.issubdtype(
        observations.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("x must be a real numeric array")
    if not jnp.issubdtype(observations.dtype, jnp.floating):
        target_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        observations = observations.astype(target_dtype, copy=False)
    elif observations.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        observations = observations.astype(jnp.float32, copy=False)
    _validate_eager(
        jnp.all(jnp.isfinite(observations)),
        "x must contain only finite values",
    )
    centered = observations - jnp.mean(observations, axis=0)
    _validate_eager(
        jnp.all(jnp.sum(jnp.square(centered), axis=0) > 0),
        "x must not contain constant columns",
    )
    return _pairwise_jeffreys_bayes_factors_unchecked(observations)


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        if minimum == 3:
            raise ValueError(f"{name} must be at least three")
        raise ValueError(f"{name} must be positive")


def _validate_scalar(name: str, value: Array) -> None:
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    _validate_eager(jnp.isfinite(value), f"{name} must be finite")


def _validate_typed_scalar_key(key: Array) -> None:
    if not isinstance(key, jax.Array) or not jnp.issubdtype(
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError("key must be a typed JAX key from jax.random.key")
    if key.ndim != 0:
        raise ValueError("key must be a scalar key; pass one unsplit master key")


def _type7_quantile_nonnegative(values: Array, probability: Array) -> Array:
    """Return R's type-7 quantile without turning equal infinities into NaN."""
    ordered = jnp.sort(values)
    position = (values.shape[0] - 1) * probability
    lower_index = jnp.floor(position).astype(jnp.int32)
    upper_index = jnp.ceil(position).astype(jnp.int32)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    weight = position - jnp.floor(position)
    interpolated = lower + weight * (upper - lower)
    return jnp.where(
        (lower_index == upper_index) | (lower == upper), lower, interpolated
    )


@partial(
    jax.jit,
    static_argnames=("n_observations", "n_simulations", "dtype_name"),
)
def _estimate_fnr_cutoff_unchecked(
    key: Array,
    correlation: Array,
    false_negative_rate: Array,
    *,
    n_observations: int,
    n_simulations: int,
    dtype_name: str,
) -> Array:
    dtype = jnp.dtype(dtype_name)
    normal_draws = jax.random.normal(
        key,
        (n_simulations, n_observations, 2),
        dtype=dtype,
    )
    first = normal_draws[..., 0]
    second = correlation * first + jnp.sqrt(1.0 - correlation**2) * normal_draws[..., 1]
    first = first - jnp.mean(first, axis=1, keepdims=True)
    second = second - jnp.mean(second, axis=1, keepdims=True)
    sample_correlations = jnp.sum(first * second, axis=1) / jnp.sqrt(
        jnp.sum(jnp.square(first), axis=1) * jnp.sum(jnp.square(second), axis=1)
    )
    sample_correlations = jnp.where(
        jnp.abs(correlation) == 1.0,
        correlation,
        sample_correlations,
    )
    scores = _jeffreys_bayes_factor_from_correlation(
        sample_correlations,
        n_observations=n_observations,
    )
    return _type7_quantile_nonnegative(scores, false_negative_rate)


def estimate_fnr_cutoff(
    key: Array,
    *,
    n_observations: int,
    correlation: float | Array = 0.25,
    false_negative_rate: float | Array = 0.05,
    n_simulations: int = 1000,
    dtype: str = "float64",
) -> Array:
    """Estimate the upstream FNR cutoff with pure-JAX simulations.

    This follows ``bspcov::select_cutoff`` statistically while using the
    explicit JAX key supplied by the caller. JAX and R use different random
    number generators, so identical seeds are not expected to produce
    bitwise-identical cutoffs.
    """
    _validate_typed_scalar_key(key)
    _validate_integer("n_observations", n_observations, minimum=3)
    _validate_integer("n_simulations", n_simulations, minimum=1)
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if dtype == "float64" and not jax.config.x64_enabled:
        raise RuntimeError(
            "dtype='float64' requires JAX X64 mode. Set JAX_ENABLE_X64=1 "
            "before starting Python."
        )
    jax_dtype = jnp.dtype(dtype)
    correlation_value = jnp.asarray(correlation, dtype=jax_dtype)
    false_negative_rate_value = jnp.asarray(
        false_negative_rate,
        dtype=jax_dtype,
    )
    _validate_scalar("correlation", correlation_value)
    _validate_scalar("false_negative_rate", false_negative_rate_value)
    _validate_eager(
        (correlation_value >= -1.0) & (correlation_value <= 1.0),
        "correlation must be between -1 and 1 (inclusive)",
    )
    _validate_eager(
        (false_negative_rate_value >= 0.0) & (false_negative_rate_value <= 1.0),
        "false_negative_rate must be between zero and one",
    )
    target = next(iter(key.devices()))
    return cast(
        Array,
        _estimate_fnr_cutoff_unchecked(
            key,
            jax.device_put(correlation_value, target),
            jax.device_put(false_negative_rate_value, target),
            n_observations=n_observations,
            n_simulations=n_simulations,
            dtype_name=dtype,
        ),
    )


def _fnr_screening_mask_unchecked(score_matrix: Array, cutoff: Array) -> Array:
    """Build an FNR active mask without validation for compiled package code."""
    lower_active_mask = jnp.tril(score_matrix > cutoff, k=-1)
    return lower_active_mask | lower_active_mask.T


def fnr_screening_mask(score_matrix: Array, cutoff: float | Array) -> Array:
    """Validate scores and build the deterministic FNR active-edge mask.

    This host-level function matches ``bspcov`` 1.0.3 ``BayesCGM.SS``: only
    lower-triangular pairwise scores are authoritative, ``True`` means that an
    off-diagonal edge is retained for sampling, and retention requires a score
    strictly greater than ``cutoff``. The diagonal is always ``False``.

    This function deliberately rejects use under :func:`jax.jit` because its
    value checks require concrete host inputs. Package code that has already
    validated its inputs may compile the private unchecked kernel.

    Use :func:`pairwise_jeffreys_bayes_factors` and
    :func:`estimate_fnr_cutoff` to compute the inputs.
    """
    scores = jnp.asarray(score_matrix)
    _reject_tracer(scores, "fnr_screening_mask")
    dimension = _validate_pairwise_shape(scores)
    lower_indices = jnp.tril_indices(dimension, k=-1)
    lower_scores = scores[lower_indices]
    cutoff_value = jnp.asarray(cutoff)
    _reject_tracer(cutoff_value, "fnr_screening_mask")
    if cutoff_value.ndim != 0:
        raise ValueError("cutoff must be a scalar")
    _validate_eager(
        jnp.all(~jnp.isnan(lower_scores)),
        "score_matrix lower triangle must not contain NaN values",
    )
    _validate_eager(
        jnp.all(lower_scores >= 0),
        "score_matrix lower triangle must contain non-negative values",
    )
    _validate_eager(~jnp.isnan(cutoff_value), "cutoff must not be NaN")
    _validate_eager(cutoff_value >= 0, "cutoff must be non-negative")

    return _fnr_screening_mask_unchecked(scores, cutoff_value)


def _correlation_screening_mask_unchecked(
    x: Array,
    retained_fraction: Array,
) -> Array:
    """Build a correlation active mask without validation for compiled code."""
    centered = x - jnp.mean(x, axis=0)
    sums_of_squares = jnp.sum(jnp.square(centered), axis=0)
    scale = jnp.sqrt(jnp.outer(sums_of_squares, sums_of_squares))
    correlations = (centered.T @ centered) / scale
    dimension = x.shape[1]
    lower_indices = jnp.tril_indices(dimension, k=-1)
    absolute_lower = jnp.abs(correlations[lower_indices])
    threshold = jnp.quantile(
        absolute_lower,
        1.0 - retained_fraction,
        method="linear",
    )
    lower_active_mask = jnp.tril(jnp.abs(correlations) > threshold, k=-1)
    return lower_active_mask | lower_active_mask.T


def correlation_screening_mask(
    x: Array,
    retained_fraction: float | Array = 0.2,
) -> Array:
    """Validate data and build the upstream correlation active-edge mask.

    ``retained_fraction`` follows the upstream ``thr`` convention: it controls
    the upper fraction of absolute pairwise sample correlations considered for
    retention; it is not an absolute correlation threshold. ``True`` means an
    off-diagonal edge is retained for sampling, while the diagonal is always
    ``False``. The cutoff uses R's type-7 linear quantile and a strict
    comparison.

    This function deliberately rejects use under :func:`jax.jit` because its
    value checks require concrete host inputs. Package code that has already
    validated its inputs may compile the private unchecked kernel.
    """
    observations = jnp.asarray(x)
    _reject_tracer(observations, "correlation_screening_mask")
    if observations.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    n_observations, dimension = observations.shape
    if n_observations < 2:
        raise ValueError("x must contain at least two observations")
    if dimension < 2:
        raise ValueError("x must contain at least two variables")

    retained = jnp.asarray(retained_fraction)
    _reject_tracer(retained, "correlation_screening_mask")
    if retained.ndim != 0:
        raise ValueError("retained_fraction must be a scalar")
    _validate_eager(
        jnp.all(jnp.isfinite(observations)),
        "x must contain only finite values",
    )
    _validate_eager(jnp.isfinite(retained), "retained_fraction must be finite")
    _validate_eager(
        (retained >= 0) & (retained <= 1),
        "retained_fraction must be between zero and one",
    )

    centered = observations - jnp.mean(observations, axis=0)
    sums_of_squares = jnp.sum(jnp.square(centered), axis=0)
    _validate_eager(
        jnp.all(sums_of_squares > 0),
        "x must not contain constant columns",
    )

    return _correlation_screening_mask_unchecked(observations, retained)
