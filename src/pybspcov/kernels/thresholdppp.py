"""Pure JAX post-processing kernels for thresholded PPP covariances."""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from pybspcov.kernels.bandppp import sample_inverse_wishart
from pybspcov.kernels.bm import pack_lower_triangle_column_major

type ThresholdMethod = Literal["hard", "soft"]


def threshold_and_adjust_covariance(
    covariance: Array,
    *,
    threshold: Array,
    method: ThresholdMethod,
    epsilon: Array,
) -> tuple[Array, Array]:
    """Threshold off-diagonal entries and enforce an eigenvalue floor."""
    dimension = covariance.shape[-1]
    diagonal_mask = jnp.eye(dimension, dtype=jnp.bool_)
    if method == "hard":
        processed = jnp.where(jnp.abs(covariance) >= threshold, covariance, 0.0)
    else:
        processed = jnp.sign(covariance) * jnp.maximum(
            jnp.abs(covariance) - threshold,
            0.0,
        )
    processed = jnp.where(diagonal_mask, covariance, processed)
    minimum_eigenvalue = jnp.linalg.eigvalsh(processed)[0]
    diagonal_shift = jnp.maximum(epsilon - minimum_eigenvalue, 0.0)
    adjusted = processed + diagonal_shift * jnp.eye(
        dimension,
        dtype=covariance.dtype,
    )
    return adjusted, minimum_eigenvalue < epsilon


def sample_thresholdppp_chains(
    keys: Array,
    posterior_scale: Array,
    posterior_degrees_of_freedom: Array,
    threshold: Array,
    epsilon: Array,
    *,
    method: ThresholdMethod,
    n_samples: int,
) -> tuple[Array, Array]:
    """Draw and post-process independent threshold PPP chains."""

    def sample_chain(key: Array) -> tuple[Array, Array]:
        initial_draws = sample_inverse_wishart(
            key,
            degrees_of_freedom=posterior_degrees_of_freedom,
            scale=posterior_scale,
            n_samples=n_samples,
        )
        adjusted_draws, adjusted = jax.vmap(
            lambda covariance: threshold_and_adjust_covariance(
                covariance,
                threshold=threshold,
                method=method,
                epsilon=epsilon,
            )
        )(initial_draws)
        return pack_lower_triangle_column_major(adjusted_draws), adjusted

    return jax.vmap(sample_chain)(keys)
