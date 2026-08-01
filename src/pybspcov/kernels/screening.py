"""Fixed-shape screening masks for screened beta-mixture kernels.

The public functions validate concrete host inputs before dispatching to pure
JAX kernels. Compiled package code must validate first and then call the
explicitly unchecked private kernels.
"""

import jax
import jax.numpy as jnp
from jax import Array


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

    The stochastic FNR cutoff and Bayes-factor scores must be computed
    separately.
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
        jnp.all(jnp.isfinite(lower_scores)),
        "score_matrix lower triangle must contain finite values",
    )
    _validate_eager(
        jnp.all(lower_scores >= 0),
        "score_matrix lower triangle must contain non-negative values",
    )
    _validate_eager(jnp.isfinite(cutoff_value), "cutoff must be finite")
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
