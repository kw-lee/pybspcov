"""Fixed-shape screening masks for screened beta-mixture kernels.

The deterministic rules mirror ``bspcov`` 1.0.3. FNR cutoff selection and
Jeffreys Bayes-factor score generation are intentionally outside this module.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def _eagerly_validate(predicate: Callable[[], Array], message: str) -> None:
    """Raise for an invalid eager value while remaining usable under ``jit``."""
    try:
        valid = bool(predicate())
    except jax.errors.TracerBoolConversionError:
        return
    if not valid:
        raise ValueError(message)


def _validate_pairwise_shape(values: Array) -> int:
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("score_matrix must be a square two-dimensional array")
    dimension = values.shape[0]
    if dimension < 2:
        raise ValueError("score_matrix must contain at least two variables")
    return dimension


def fnr_screening_mask(score_matrix: Array, cutoff: float | Array) -> Array:
    """Build the deterministic FNR active-edge mask from precomputed BF scores.

    This matches ``bspcov`` 1.0.3 ``BayesCGM.SS``: only lower-triangular
    pairwise scores are authoritative, ``True`` means that an off-diagonal edge
    is retained for sampling, and retention requires a score strictly greater
    than ``cutoff``. The diagonal is always ``False``. The
    stochastic FNR cutoff and Bayes-factor scores must be computed separately.
    """
    scores = jnp.asarray(score_matrix)
    dimension = _validate_pairwise_shape(scores)
    lower_indices = jnp.tril_indices(dimension, k=-1)
    lower_scores = scores[lower_indices]
    cutoff_value = jnp.asarray(cutoff)
    if cutoff_value.ndim != 0:
        raise ValueError("cutoff must be a scalar")
    _eagerly_validate(
        lambda: jnp.all(jnp.isfinite(lower_scores)),
        "score_matrix lower triangle must contain finite values",
    )
    _eagerly_validate(
        lambda: jnp.all(lower_scores >= 0),
        "score_matrix lower triangle must contain non-negative values",
    )
    _eagerly_validate(
        lambda: jnp.isfinite(cutoff_value),
        "cutoff must be finite",
    )
    _eagerly_validate(
        lambda: cutoff_value >= 0,
        "cutoff must be non-negative",
    )

    lower_active_mask = jnp.tril(scores > cutoff_value, k=-1)
    return lower_active_mask | lower_active_mask.T


def correlation_screening_mask(
    x: Array,
    retained_fraction: float | Array = 0.2,
) -> Array:
    """Build the upstream correlation active-edge mask for raw or centered data.

    ``retained_fraction`` follows the upstream ``thr`` convention: it controls
    the upper fraction of absolute pairwise sample correlations considered for
    retention; it is not an absolute correlation threshold. ``True`` means an
    off-diagonal edge is retained for sampling, while the diagonal is always
    ``False``. The cutoff uses R's type-7 linear quantile and a strict
    comparison.
    """
    observations = jnp.asarray(x)
    if observations.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    n_observations, dimension = observations.shape
    if n_observations < 2:
        raise ValueError("x must contain at least two observations")
    if dimension < 2:
        raise ValueError("x must contain at least two variables")

    retained = jnp.asarray(retained_fraction)
    if retained.ndim != 0:
        raise ValueError("retained_fraction must be a scalar")
    _eagerly_validate(
        lambda: jnp.all(jnp.isfinite(observations)),
        "x must contain only finite values",
    )
    _eagerly_validate(
        lambda: jnp.isfinite(retained),
        "retained_fraction must be finite",
    )
    _eagerly_validate(
        lambda: (retained >= 0) & (retained <= 1),
        "retained_fraction must be between zero and one",
    )

    centered = observations - jnp.mean(observations, axis=0)
    sums_of_squares = jnp.sum(jnp.square(centered), axis=0)
    _eagerly_validate(
        lambda: jnp.all(sums_of_squares > 0),
        "x must not contain constant columns",
    )

    scale = jnp.sqrt(jnp.outer(sums_of_squares, sums_of_squares))
    correlations = (centered.T @ centered) / scale
    lower_indices = jnp.tril_indices(dimension, k=-1)
    absolute_lower = jnp.abs(correlations[lower_indices])
    threshold = jnp.quantile(
        absolute_lower,
        1.0 - retained,
        method="linear",
    )
    lower_active_mask = jnp.tril(jnp.abs(correlations) > threshold, k=-1)
    return lower_active_mask | lower_active_mask.T
