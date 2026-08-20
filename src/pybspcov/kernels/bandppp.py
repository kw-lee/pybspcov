"""Pure JAX sampling and post-processing kernels for BandPPP."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from pybspcov.kernels.bm import pack_lower_triangle_column_major


def _sample_inverse_wishart(
    key: Array,
    *,
    degrees_of_freedom: Array,
    scale_cholesky: Array,
) -> Array:
    dimension = scale_cholesky.shape[-1]
    normal_key, gamma_key = jax.random.split(key)
    bartlett = jnp.tril(
        jax.random.normal(
            normal_key,
            shape=(dimension, dimension),
            dtype=scale_cholesky.dtype,
        ),
        k=-1,
    )
    chi_squared_degrees = degrees_of_freedom - jnp.arange(
        dimension,
        dtype=scale_cholesky.dtype,
    )
    diagonal = jnp.sqrt(
        2.0
        * jax.random.gamma(
            gamma_key,
            chi_squared_degrees / 2.0,
            dtype=scale_cholesky.dtype,
        )
    )
    bartlett = bartlett + jnp.diag(diagonal)
    factor = solve_triangular(
        bartlett,
        scale_cholesky.T,
        lower=True,
    )
    return factor.T @ factor


def sample_inverse_wishart(
    key: Array,
    *,
    degrees_of_freedom: Array,
    scale: Array,
    n_samples: int,
) -> Array:
    """Draw inverse-Wishart matrices with the Bartlett decomposition."""
    scale_cholesky = jnp.linalg.cholesky(scale)
    keys = jax.random.split(key, n_samples)
    return jax.vmap(
        lambda sample_key: _sample_inverse_wishart(
            sample_key,
            degrees_of_freedom=degrees_of_freedom,
            scale_cholesky=scale_cholesky,
        )
    )(keys)


def band_and_adjust_covariance(
    covariance: Array,
    *,
    bandwidth: int | Array,
    epsilon: Array,
) -> tuple[Array, Array]:
    """Band one covariance matrix and enforce its eigenvalue floor."""
    dimension = covariance.shape[-1]
    indices = jnp.arange(dimension)
    band_mask = jnp.abs(indices[:, None] - indices[None, :]) <= bandwidth
    banded = jnp.where(band_mask, covariance, 0.0)
    minimum_eigenvalue = jnp.linalg.eigvalsh(banded)[0]
    diagonal_shift = jnp.maximum(epsilon - minimum_eigenvalue, 0.0)
    adjusted = banded + diagonal_shift * jnp.eye(
        dimension,
        dtype=covariance.dtype,
    )
    return adjusted, minimum_eigenvalue < epsilon


def sample_bandppp_chains(
    keys: Array,
    posterior_scale: Array,
    posterior_degrees_of_freedom: Array,
    bandwidth: Array,
    epsilon: Array,
    *,
    n_samples: int,
) -> tuple[Array, Array]:
    """Draw and post-process independent BandPPP chains."""

    def sample_chain(key: Array) -> tuple[Array, Array]:
        initial_draws = sample_inverse_wishart(
            key,
            degrees_of_freedom=posterior_degrees_of_freedom,
            scale=posterior_scale,
            n_samples=n_samples,
        )
        adjusted_draws, adjusted = jax.vmap(
            lambda covariance: band_and_adjust_covariance(
                covariance,
                bandwidth=bandwidth,
                epsilon=epsilon,
            )
        )(initial_draws)
        return pack_lower_triangle_column_major(adjusted_draws), adjusted

    return jax.vmap(sample_chain)(keys)
