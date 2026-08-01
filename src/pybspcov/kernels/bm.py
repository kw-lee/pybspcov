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


class BMColumnParameters(NamedTuple):
    """Deterministic conditional parameters for one BM column update."""

    conditional_precision: Array
    conditional_scatter: Array
    quadratic: Array
    gamma_lambda: Array
    gamma_chi: Array
    gamma_psi: Array
    beta_precision: Array
    beta_mean: Array


class BMChainResult(NamedTuple):
    """Retained BM draws, sweep statuses, and the final chain state."""

    final_state: BMState
    covariance: Array
    phi: Array
    accepted: Array


class _BMColumnMoments(NamedTuple):
    conditional_precision: Array
    conditional_scatter: Array
    quadratic: Array
    gamma_lambda: Array
    gamma_chi: Array
    gamma_psi: Array


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


def _bm_column_moments(
    covariance: Array,
    precision: Array,
    scatter: Array,
    column: Array,
    other_indices: Array,
    n_observations: Array,
    diagonal_rate: Array,
) -> _BMColumnMoments:
    precision_block = precision[jnp.ix_(other_indices, other_indices)]
    precision_cross = precision[other_indices, column]
    conditional_precision = (
        precision_block
        - jnp.outer(precision_cross, precision_cross) / precision[column, column]
    )
    scatter_block = scatter[jnp.ix_(other_indices, other_indices)]
    scatter_cross = scatter[other_indices, column]
    conditional_scatter = conditional_precision @ scatter_cross
    quadratic = conditional_precision @ scatter_block @ conditional_precision
    beta = covariance[other_indices, column]
    gamma_chi = (
        beta @ quadratic @ beta
        - 2.0 * beta @ conditional_scatter
        + scatter[column, column]
    )
    return _BMColumnMoments(
        conditional_precision=conditional_precision,
        conditional_scatter=conditional_scatter,
        quadratic=quadratic,
        gamma_lambda=1.0 - n_observations / 2.0,
        gamma_chi=gamma_chi,
        gamma_psi=diagonal_rate,
    )


def _bm_beta_parameters(
    moments: _BMColumnMoments,
    tau: Array,
    other_indices: Array,
    column: Array,
    diagonal_rate: Array,
    gamma: Array,
) -> tuple[Array, Array]:
    beta_precision = (
        moments.quadratic / gamma
        + jnp.diag(1.0 / tau[other_indices, column])
        + diagonal_rate * moments.conditional_precision
    )
    beta_precision = 0.5 * (beta_precision + beta_precision.T)
    beta_mean = jnp.linalg.solve(beta_precision, moments.conditional_scatter) / gamma
    return beta_precision, beta_mean


def bm_column_parameters(
    *,
    covariance: Array,
    precision: Array,
    scatter: Array,
    tau: Array,
    column: Array,
    other_indices: Array,
    n_observations: Array,
    diagonal_rate: Array,
    gamma: Array,
) -> BMColumnParameters:
    """Return the R-compatible conditionals for one blocked BM update."""
    moments = _bm_column_moments(
        covariance,
        precision,
        scatter,
        column,
        other_indices,
        n_observations,
        diagonal_rate,
    )
    beta_precision, beta_mean = _bm_beta_parameters(
        moments,
        tau,
        other_indices,
        column,
        diagonal_rate,
        gamma,
    )
    return BMColumnParameters(
        conditional_precision=moments.conditional_precision,
        conditional_scatter=moments.conditional_scatter,
        quadratic=moments.quadratic,
        gamma_lambda=moments.gamma_lambda,
        gamma_chi=moments.gamma_chi,
        gamma_psi=moments.gamma_psi,
        beta_precision=beta_precision,
        beta_mean=beta_mean,
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
        moments = _bm_column_moments(
            current.covariance,
            current.precision,
            scatter,
            jnp.asarray(column),
            indices,
            n_observations,
            diagonal_rate,
        )
        gamma_draw = sample_gig(
            gamma_key,
            moments.gamma_lambda,
            jnp.maximum(
                moments.gamma_chi,
                jnp.asarray(1e-12, dtype=dtype),
            ),
            moments.gamma_psi,
        )
        gamma = jnp.where(
            gamma_draw.accepted,
            gamma_draw.value,
            jnp.asarray(1.0, dtype=dtype),
        )
        beta_precision, beta_mean = _bm_beta_parameters(
            moments,
            current.tau,
            indices,
            jnp.asarray(column),
            diagonal_rate,
            gamma,
        )
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
            sweep_accepted & gamma_draw.accepted & jnp.all(phi_draws.accepted) & finite
        )
        return updated, current_key, accepted

    updated, _, accepted = jax.lax.fori_loop(
        0,
        dimension,
        update_column,
        (state, key, jnp.asarray(True)),
    )
    return BMSweepResult(state=updated, accepted=accepted)


def sample_bm_chain(
    key: Array,
    state: BMState,
    scatter: Array,
    other_indices: Array,
    n_observations: Array,
    a: Array,
    b: Array,
    diagonal_rate: Array,
    tau1sq: Array,
    *,
    burnin: int,
    n_samples: int,
) -> BMChainResult:
    """Run sequential BM sweeps and retain draws after burn-in."""
    if burnin < 0:
        raise ValueError("burnin must be non-negative")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    sweep_keys = jax.random.split(key, burnin + n_samples)

    def sweep(
        current: BMState,
        sweep_key: Array,
    ) -> tuple[BMState, tuple[Array, Array, Array]]:
        result = bm_sweep(
            sweep_key,
            current,
            scatter,
            other_indices,
            n_observations,
            a,
            b,
            diagonal_rate,
            tau1sq,
        )
        return result.state, (
            result.state.covariance,
            result.state.phi,
            result.accepted,
        )

    final_state, (covariance, phi, accepted) = jax.lax.scan(
        sweep,
        state,
        sweep_keys,
    )
    return BMChainResult(
        final_state=final_state,
        covariance=covariance[burnin:],
        phi=phi[burnin:],
        accepted=accepted,
    )
