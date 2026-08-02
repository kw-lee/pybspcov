"""Fixed-shape masked-dense screened beta-mixture kernels."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from pybspcov.kernels.bm import (
    BMState,
    BMSweepResult,
    _bm_column_moments,
    initialize_bm_state,
)
from pybspcov.kernels.covariance import update_covariance_column
from pybspcov.sampling.gig import _sample_gig_batch, sample_gig


class SBMColumnParameters(NamedTuple):
    """Padded conditional parameters for one screened SBM column update."""

    active: Array
    active_count: Array
    conditional_precision: Array
    conditional_scatter: Array
    quadratic: Array
    gamma_lambda: Array
    gamma_chi: Array
    gamma_psi: Array
    beta_precision: Array
    beta_mean: Array


def sbm_column_parameters(
    *,
    covariance: Array,
    precision: Array,
    scatter: Array,
    tau: Array,
    active_mask: Array,
    column: Array,
    other_indices: Array,
    n_observations: Array,
    diagonal_rate: Array,
    gamma: Array,
) -> SBMColumnParameters:
    """Return R-compatible SBM conditionals in fixed ``(p - 1)`` shapes.

    Active rows and columns contain the same reduced system used by
    ``bspcov::sbm.covest``. Screened positions are decoupled with an identity
    block, which keeps the solve nonsingular and produces exact zero padded
    coefficients without data-dependent array shapes.
    """
    moments = _sbm_column_moments(
        covariance,
        precision,
        scatter,
        active_mask,
        column,
        other_indices,
        n_observations,
        diagonal_rate,
    )

    unmasked_beta_precision = (
        moments.quadratic / gamma
        + jnp.diag(1.0 / tau[other_indices, column])
        + diagonal_rate * moments.conditional_precision
    )
    active_outer = moments.active[:, None] & moments.active[None, :]
    beta_precision = jnp.where(active_outer, unmasked_beta_precision, 0.0)
    beta_precision = beta_precision + jnp.diag(
        (~moments.active).astype(covariance.dtype)
    )
    beta_precision = 0.5 * (beta_precision + beta_precision.T)
    beta_mean = jnp.linalg.solve(beta_precision, moments.conditional_scatter) / gamma

    return SBMColumnParameters(
        active=moments.active,
        active_count=moments.active_count,
        conditional_precision=moments.conditional_precision,
        conditional_scatter=moments.conditional_scatter,
        quadratic=moments.quadratic,
        gamma_lambda=moments.gamma_lambda,
        gamma_chi=moments.gamma_chi,
        gamma_psi=moments.gamma_psi,
        beta_precision=beta_precision,
        beta_mean=beta_mean,
    )


class _SBMColumnMoments(NamedTuple):
    active: Array
    active_count: Array
    conditional_precision: Array
    conditional_scatter: Array
    quadratic: Array
    gamma_lambda: Array
    gamma_chi: Array
    gamma_psi: Array


def _sbm_column_moments(
    covariance: Array,
    precision: Array,
    scatter: Array,
    active_mask: Array,
    column: Array,
    other_indices: Array,
    n_observations: Array,
    diagonal_rate: Array,
) -> _SBMColumnMoments:
    moments = _bm_column_moments(
        covariance,
        precision,
        scatter,
        column,
        other_indices,
        n_observations,
        diagonal_rate,
    )
    active = active_mask[other_indices, column]
    active_outer = active[:, None] & active[None, :]
    conditional_scatter = jnp.where(active, moments.conditional_scatter, 0.0)
    quadratic = jnp.where(active_outer, moments.quadratic, 0.0)
    beta = jnp.where(active, covariance[other_indices, column], 0.0)
    gamma_chi = (
        beta @ quadratic @ beta
        - 2.0 * beta @ conditional_scatter
        + scatter[column, column]
    )
    return _SBMColumnMoments(
        active=active,
        active_count=jnp.sum(active, dtype=jnp.int32),
        conditional_precision=moments.conditional_precision,
        conditional_scatter=conditional_scatter,
        quadratic=quadratic,
        gamma_lambda=moments.gamma_lambda,
        gamma_chi=gamma_chi,
        gamma_psi=moments.gamma_psi,
    )


def _sbm_beta_parameters(
    moments: _SBMColumnMoments,
    tau: Array,
    other_indices: Array,
    column: Array,
    diagonal_rate: Array,
    gamma: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    active_outer = moments.active[:, None] & moments.active[None, :]
    unmasked_beta_precision = (
        moments.quadratic / gamma
        + jnp.diag(1.0 / tau[other_indices, column])
        + diagonal_rate * moments.conditional_precision
    )
    beta_precision = jnp.where(active_outer, unmasked_beta_precision, 0.0)
    beta_precision = beta_precision + jnp.diag((~moments.active).astype(dtype))
    beta_precision = 0.5 * (beta_precision + beta_precision.T)
    beta_mean = jnp.linalg.solve(beta_precision, moments.conditional_scatter) / gamma
    return beta_precision, beta_mean


def initialize_sbm_state(
    covariance: Array,
    tau1sq: Array,
    active_mask: Array,
) -> BMState:
    """Construct an SBM state after enforcing the fixed covariance support."""
    dimension = covariance.shape[0]
    identity = jnp.eye(dimension, dtype=covariance.dtype)
    supported = active_mask | identity.astype(jnp.bool_)
    screened_covariance = jnp.where(supported, covariance, 0.0)
    minimum_eigenvalue = jnp.linalg.eigvalsh(screened_covariance)[0]
    jitter = jnp.where(
        minimum_eigenvalue <= jnp.asarray(1e-15, dtype=covariance.dtype),
        -minimum_eigenvalue + jnp.asarray(0.001, dtype=covariance.dtype),
        jnp.asarray(0.0, dtype=covariance.dtype),
    )
    screened_covariance = screened_covariance + jitter * identity
    return initialize_bm_state(screened_covariance, tau1sq)


def sbm_sweep(
    key: Array,
    state: BMState,
    scatter: Array,
    other_indices: Array,
    n_observations: Array,
    a: Array,
    b: Array,
    diagonal_rate: Array,
    tau1sq: Array,
    active_mask: Array,
) -> BMSweepResult:
    """Run one fixed-shape masked-dense SBM Gibbs sweep."""
    dimension = scatter.shape[0]
    padded_count = dimension - 1
    dtype = state.covariance.dtype

    def update_column(
        column: int,
        carry: tuple[BMState, Array, Array],
    ) -> tuple[BMState, Array, Array]:
        current, current_key, sweep_accepted = carry
        current_key, gamma_key, beta_key, phi_key, psi_key = jax.random.split(
            current_key, 5
        )
        column_array = jnp.asarray(column)
        indices = other_indices[column]
        moments = _sbm_column_moments(
            current.covariance,
            current.precision,
            scatter,
            active_mask,
            column_array,
            indices,
            n_observations,
            diagonal_rate,
        )
        gamma_draw = sample_gig(
            gamma_key,
            moments.gamma_lambda,
            jnp.maximum(moments.gamma_chi, jnp.asarray(1e-12, dtype=dtype)),
            moments.gamma_psi,
        )
        gamma = jnp.where(
            gamma_draw.accepted,
            gamma_draw.value,
            jnp.asarray(1.0, dtype=dtype),
        )
        beta_precision, beta_mean = _sbm_beta_parameters(
            moments,
            current.tau,
            indices,
            column_array,
            diagonal_rate,
            gamma,
            dtype,
        )
        beta_cholesky = jnp.linalg.cholesky(beta_precision)
        beta_noise = jnp.linalg.solve(
            beta_cholesky.T,
            jax.random.normal(beta_key, (padded_count,), dtype=dtype),
        )
        beta = jnp.where(moments.active, beta_mean + beta_noise, 0.0)
        covariance, precision = update_covariance_column(
            current.covariance,
            current.precision,
            column_array,
            indices,
            beta,
            gamma,
        )

        phi_keys = jax.random.split(phi_key, padded_count)
        phi_chi = jnp.where(
            moments.active,
            jnp.maximum(jnp.square(beta) / tau1sq, 1e-6),
            jnp.asarray(-1.0, dtype=dtype),
        )
        phi_psi = jnp.where(
            moments.active,
            2.0 * current.psi[indices, column],
            jnp.asarray(-1.0, dtype=dtype),
        )
        phi_draws = _sample_gig_batch(
            phi_keys,
            a - 0.5,
            phi_chi,
            phi_psi,
        )
        phi_accepted = moments.active & phi_draws.accepted
        phi_values = jnp.where(
            phi_accepted,
            phi_draws.value,
            current.phi[indices, column],
        )
        psi_draws = jax.random.gamma(
            psi_key,
            a + b,
            shape=(padded_count,),
            dtype=dtype,
        ) / (phi_values + 1.0)
        psi_values = jnp.where(
            phi_accepted,
            psi_draws,
            current.psi[indices, column],
        )
        tau_values = jnp.where(
            phi_accepted,
            phi_values * tau1sq,
            current.tau[indices, column],
        )

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
            & jnp.all((~moments.active) | phi_draws.accepted)
            & finite
        )
        return updated, current_key, accepted

    updated, _, accepted = jax.lax.fori_loop(
        0,
        dimension,
        update_column,
        (state, key, jnp.asarray(True)),
    )
    committed = jax.lax.cond(
        accepted,
        lambda _: updated,
        lambda _: state,
        operand=None,
    )
    return BMSweepResult(state=committed, accepted=accepted)


class SBMChainResult(NamedTuple):
    """Retained masked-dense SBM draws and the final chain state."""

    final_state: BMState
    covariance: Array
    phi: Array
    accepted: Array


def sample_sbm_chain(
    key: Array,
    state: BMState,
    scatter: Array,
    other_indices: Array,
    n_observations: Array,
    a: Array,
    b: Array,
    diagonal_rate: Array,
    tau1sq: Array,
    active_mask: Array,
    *,
    burnin: int,
    n_samples: int,
) -> SBMChainResult:
    """Run sequential masked-dense SBM sweeps and retain post-burn-in draws."""
    if burnin < 0:
        raise ValueError("burnin must be non-negative")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    sweep_keys = jax.random.split(key, burnin + n_samples)

    def sweep(
        current: BMState,
        sweep_key: Array,
    ) -> tuple[BMState, tuple[Array, Array, Array]]:
        result = sbm_sweep(
            sweep_key,
            current,
            scatter,
            other_indices,
            n_observations,
            a,
            b,
            diagonal_rate,
            tau1sq,
            active_mask,
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
    return SBMChainResult(
        final_state=final_state,
        covariance=covariance[burnin:],
        phi=phi[burnin:],
        accepted=accepted,
    )


def validate_sbm_active_mask(
    active_mask: Array,
    *,
    dimension: int,
) -> Array:
    """Validate a concrete symmetric SBM support mask before compilation."""
    mask = jnp.asarray(active_mask)
    if mask.ndim != 2 or mask.shape != (dimension, dimension):
        raise ValueError(f"active_mask must have shape ({dimension}, {dimension})")
    if mask.dtype != jnp.bool_:
        raise TypeError("active_mask must contain boolean values")
    if isinstance(mask, jax.core.Tracer):
        raise TypeError(
            "validate_sbm_active_mask requires a concrete host mask before jax.jit"
        )
    if not bool(jnp.array_equal(mask, mask.T)):
        raise ValueError("active_mask must be symmetric")
    if bool(jnp.any(jnp.diag(mask))):
        raise ValueError("active_mask diagonal must be false")
    return mask
