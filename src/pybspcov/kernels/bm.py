"""Pure JAX beta-mixture covariance Gibbs updates."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from pybspcov.kernels.covariance import update_covariance_column
from pybspcov.sampling.gig import sample_gig


class BMState(NamedTuple):
    """Immutable state of one beta-mixture covariance chain."""

    covariance: Array
    precision: Array
    phi: Array
    psi: Array
    tau: Array


class BMSweepResult(NamedTuple):
    """Updated BM state and aggregate sampler status."""

    state: BMState
    accepted: Array


def initialize_bm_state(covariance: Array, tau1sq: Array) -> BMState:
    """Construct the R-compatible initial BM chain state."""
    ones = jnp.ones_like(covariance)
    tau = jnp.full(covariance.shape, tau1sq, dtype=covariance.dtype)
    return BMState(
        covariance=covariance,
        precision=jnp.linalg.inv(covariance),
        phi=ones,
        psi=ones,
        tau=tau,
    )


def bm_sweep(
    key: Array,
    state: BMState,
    scatter: Array,
    other_indices: Array,
    n_observations: Array,
    a: Array,
    b: Array,
    diagonal_rate: Array,
    tau1sq: Array,
) -> BMSweepResult:
    """Run one blocked Gibbs sweep over every covariance column."""
    dimension = scatter.shape[0]
    active_count = dimension - 1
    dtype = state.covariance.dtype

    def update_column(
        column: int,
        carry: tuple[BMState, Array, Array],
    ) -> tuple[BMState, Array, Array]:
        current, current_key, sweep_accepted = carry
        current_key, gamma_key, beta_key, phi_key, psi_key = jax.random.split(
            current_key, 5
        )
        indices = other_indices[column]
        precision_block = current.precision[jnp.ix_(indices, indices)]
        precision_cross = current.precision[indices, column]
        conditional_precision = precision_block - jnp.outer(
            precision_cross, precision_cross
        ) / current.precision[column, column]
        scatter_block = scatter[jnp.ix_(indices, indices)]
        scatter_cross = scatter[indices, column]
        conditional_scatter = conditional_precision @ scatter_cross
        quadratic = conditional_precision @ scatter_block @ conditional_precision
        old_beta = current.covariance[indices, column]
        chi = (
            old_beta @ quadratic @ old_beta
            - 2.0 * old_beta @ conditional_scatter
            + scatter[column, column]
        )
        gamma_draw = sample_gig(
            gamma_key,
            1.0 - n_observations / 2.0,
            jnp.maximum(chi, jnp.asarray(1e-12, dtype=dtype)),
            diagonal_rate,
        )
        gamma = jnp.where(
            gamma_draw.accepted,
            gamma_draw.value,
            jnp.asarray(1.0, dtype=dtype),
        )
        beta_precision = (
            quadratic / gamma
            + jnp.diag(1.0 / current.tau[indices, column])
            + diagonal_rate * conditional_precision
        )
        beta_precision = 0.5 * (beta_precision + beta_precision.T)
        beta_mean = jnp.linalg.solve(beta_precision, conditional_scatter) / gamma
        beta_cholesky = jnp.linalg.cholesky(beta_precision)
        beta_noise = jnp.linalg.solve(
            beta_cholesky.T,
            jax.random.normal(beta_key, (active_count,), dtype=dtype),
        )
        beta = beta_mean + beta_noise
        covariance, precision = update_covariance_column(
            current.covariance,
            current.precision,
            jnp.asarray(column),
            indices,
            beta,
            gamma,
        )
        phi_keys = jax.random.split(phi_key, active_count)
        phi_chi = jnp.maximum(jnp.square(beta) / tau1sq, 1e-6)
        phi_draws = jax.vmap(
            lambda draw_key, draw_chi, draw_psi: sample_gig(
                draw_key,
                a - 0.5,
                draw_chi,
                2.0 * draw_psi,
            )
        )(phi_keys, phi_chi, current.psi[indices, column])
        phi_values = jnp.where(
            phi_draws.accepted,
            phi_draws.value,
            current.phi[indices, column],
        )
        psi_values = jax.random.gamma(
            psi_key,
            a + b,
            shape=(active_count,),
            dtype=dtype,
        ) / (phi_values + 1.0)
        tau_values = phi_values * tau1sq
        phi = current.phi.at[indices, column].set(phi_values)
        phi = phi.at[column, indices].set(phi_values)
        psi = current.psi.at[indices, column].set(psi_values)
        psi = psi.at[column, indices].set(psi_values)
        tau = current.tau.at[indices, column].set(tau_values)
        tau = tau.at[column, indices].set(tau_values)
        updated = BMState(covariance, precision, phi, psi, tau)
        finite = jnp.logical_and.reduce(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in updated])
        )
        accepted = (
            sweep_accepted
            & gamma_draw.accepted
            & jnp.all(phi_draws.accepted)
            & finite
        )
        return updated, current_key, accepted

    updated, _, accepted = jax.lax.fori_loop(
        0,
        dimension,
        update_column,
        (state, key, jnp.asarray(True)),
    )
    return BMSweepResult(state=updated, accepted=accepted)
