"""Covariance and precision updates used by blocked Gibbs samplers."""

import jax.numpy as jnp
from jax import Array


def update_covariance_column(
    covariance: Array,
    precision: Array,
    column: Array,
    other_indices: Array,
    beta: Array,
    gamma: Array,
) -> tuple[Array, Array]:
    """Apply one BM/SBM column update and its rank-one inverse update.

    ``gamma`` is the positive Schur complement sampled for the selected column,
    and ``beta`` contains the newly sampled off-diagonal covariance entries.
    ``other_indices`` has fixed shape ``(p - 1,)`` so the function can be used
    inside JAX-compiled loops.
    """
    block = precision[jnp.ix_(other_indices, other_indices)]
    cross = precision[other_indices, column]
    conditional_precision = block - jnp.outer(cross, cross) / precision[column, column]
    conditional_times_beta = conditional_precision @ beta

    updated_covariance = covariance.at[other_indices, column].set(beta)
    updated_covariance = updated_covariance.at[column, other_indices].set(beta)
    updated_covariance = updated_covariance.at[column, column].set(
        gamma + beta @ conditional_times_beta
    )

    updated_block = conditional_precision + jnp.outer(
        conditional_times_beta, conditional_times_beta
    ) / gamma
    updated_cross = -conditional_times_beta / gamma
    updated_precision = precision.at[jnp.ix_(other_indices, other_indices)].set(
        updated_block
    )
    updated_precision = updated_precision.at[other_indices, column].set(updated_cross)
    updated_precision = updated_precision.at[column, other_indices].set(updated_cross)
    updated_precision = updated_precision.at[column, column].set(1.0 / gamma)

    return updated_covariance, updated_precision
