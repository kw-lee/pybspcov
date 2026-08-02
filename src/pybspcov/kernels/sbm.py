"""Fixed-shape masked-dense screened beta-mixture kernels."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pybspcov.kernels.bm import (
    BMState,
    BMSweepResult,
    _bm_column_moments,
    initialize_bm_state,
)
from pybspcov.kernels.covariance import update_covariance_column
from pybspcov.sampling.gig import _sample_gig_batch, sample_gig


class SBMCompactStructure(NamedTuple):
    """Host-precomputed fixed-width active lanes for all SBM columns."""

    other_indices: Array
    active_positions: Array
    lane_mask: Array


class SBMCompactColumnParameters(NamedTuple):
    """Dense Schur complement and compact-width active beta parameters."""

    lane_mask: Array
    active_count: Array
    conditional_precision: Array
    conditional_scatter: Array
    quadratic: Array
    gamma_lambda: Array
    gamma_chi: Array
    gamma_psi: Array
    beta_precision: Array
    beta_mean: Array


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


def compact_sbm_column_parameters(
    *,
    covariance: Array,
    precision: Array,
    scatter: Array,
    tau: Array,
    column: Array,
    structure: SBMCompactStructure,
    n_observations: Array,
    diagonal_rate: Array,
    gamma: Array,
) -> SBMCompactColumnParameters:
    """Return dense Schur moments and compact-width beta conditionals."""
    other_indices = structure.other_indices[column]
    active_positions = structure.active_positions[column]
    lane_mask = structure.lane_mask[column]
    compact_indices = other_indices[active_positions]

    precision_block = precision[jnp.ix_(other_indices, other_indices)]
    precision_cross = precision[other_indices, column]
    conditional_precision = (
        precision_block
        - jnp.outer(precision_cross, precision_cross) / precision[column, column]
    )
    reduced_rows = jnp.where(
        lane_mask[:, None],
        conditional_precision[active_positions, :],
        0.0,
    )
    scatter_block = scatter[jnp.ix_(other_indices, other_indices)]
    scatter_cross = scatter[other_indices, column]
    conditional_scatter = reduced_rows @ scatter_cross
    quadratic = reduced_rows @ scatter_block @ reduced_rows.T
    beta = jnp.where(lane_mask, covariance[compact_indices, column], 0.0)
    gamma_chi = (
        beta @ quadratic @ beta
        - 2.0 * beta @ conditional_scatter
        + scatter[column, column]
    )

    lane_outer = lane_mask[:, None] & lane_mask[None, :]
    conditional_block = conditional_precision[
        jnp.ix_(active_positions, active_positions)
    ]
    unmasked_beta_precision = (
        quadratic / gamma
        + jnp.diag(1.0 / tau[compact_indices, column])
        + diagonal_rate * conditional_block
    )
    beta_precision = jnp.where(lane_outer, unmasked_beta_precision, 0.0)
    beta_precision = beta_precision + jnp.diag((~lane_mask).astype(covariance.dtype))
    beta_precision = 0.5 * (beta_precision + beta_precision.T)
    beta_mean = jnp.linalg.solve(beta_precision, conditional_scatter) / gamma
    beta_mean = jnp.where(lane_mask, beta_mean, 0.0)

    return SBMCompactColumnParameters(
        lane_mask=lane_mask,
        active_count=jnp.sum(lane_mask, dtype=jnp.int32),
        conditional_precision=conditional_precision,
        conditional_scatter=conditional_scatter,
        quadratic=quadratic,
        gamma_lambda=1.0 - n_observations / 2.0,
        gamma_chi=gamma_chi,
        gamma_psi=diagonal_rate,
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


def _update_sbm_local_scales(
    *,
    phi_key: Array,
    psi_key: Array,
    beta: Array,
    active: Array,
    active_count: Array,
    random_positions: Array,
    padded_count: int,
    current_phi: Array,
    current_psi: Array,
    current_tau: Array,
    a: Array,
    b: Array,
    tau1sq: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array]:
    """Update active local scales without executing a fully screened GIG batch."""
    def preserve_scales(_: None) -> tuple[Array, Array, Array, Array]:
        return current_phi, current_psi, current_tau, jnp.asarray(True)

    def sample_active_scales(_: None) -> tuple[Array, Array, Array, Array]:
        phi_keys = jax.random.split(phi_key, padded_count)[random_positions]
        phi_chi = jnp.where(
            active,
            jnp.maximum(jnp.square(beta) / tau1sq, 1e-6),
            jnp.asarray(-1.0, dtype=dtype),
        )
        phi_psi = jnp.where(
            active,
            2.0 * current_psi,
            jnp.asarray(-1.0, dtype=dtype),
        )
        phi_draws = _sample_gig_batch(phi_keys, a - 0.5, phi_chi, phi_psi)
        phi_accepted = active & phi_draws.accepted
        phi_values = jnp.where(phi_accepted, phi_draws.value, current_phi)
        psi_draws = jax.random.gamma(
            psi_key,
            a + b,
            shape=(padded_count,),
            dtype=dtype,
        )[random_positions]
        psi_values = jnp.where(
            phi_accepted,
            psi_draws / (phi_values + 1.0),
            current_psi,
        )
        tau_values = jnp.where(phi_accepted, phi_values * tau1sq, current_tau)
        accepted = jnp.all((~active) | phi_draws.accepted)
        return phi_values, psi_values, tau_values, accepted

    if active.shape[0] == 0:
        return preserve_scales(None)
    result: tuple[Array, Array, Array, Array] = jax.lax.cond(
        active_count == 0,
        preserve_scales,
        sample_active_scales,
        operand=None,
    )
    return result


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

        phi_values, psi_values, tau_values, scales_accepted = _update_sbm_local_scales(
            phi_key=phi_key,
            psi_key=psi_key,
            beta=beta,
            active=moments.active,
            active_count=moments.active_count,
            random_positions=jnp.arange(padded_count, dtype=jnp.int32),
            padded_count=padded_count,
            current_phi=current.phi[indices, column],
            current_psi=current.psi[indices, column],
            current_tau=current.tau[indices, column],
            a=a,
            b=b,
            tau1sq=tau1sq,
            dtype=dtype,
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
        accepted = sweep_accepted & gamma_draw.accepted & scales_accepted & finite
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


def compact_sbm_sweep(
    key: Array,
    state: BMState,
    scatter: Array,
    n_observations: Array,
    a: Array,
    b: Array,
    diagonal_rate: Array,
    tau1sq: Array,
    structure: SBMCompactStructure,
) -> BMSweepResult:
    """Run one SBM sweep using structure-owned compact active lanes."""
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
        indices = structure.other_indices[column]
        active_positions = structure.active_positions[column]
        lane_mask = structure.lane_mask[column]
        compact_indices = indices[active_positions]

        precision_block = current.precision[jnp.ix_(indices, indices)]
        precision_cross = current.precision[indices, column]
        conditional_precision = (
            precision_block
            - jnp.outer(precision_cross, precision_cross)
            / current.precision[column, column]
        )
        reduced_rows = jnp.where(
            lane_mask[:, None],
            conditional_precision[active_positions, :],
            0.0,
        )
        scatter_block = scatter[jnp.ix_(indices, indices)]
        scatter_cross = scatter[indices, column]
        conditional_scatter = reduced_rows @ scatter_cross
        quadratic = reduced_rows @ scatter_block @ reduced_rows.T
        current_beta = jnp.where(
            lane_mask,
            current.covariance[compact_indices, column],
            0.0,
        )
        gamma_chi = (
            current_beta @ quadratic @ current_beta
            - 2.0 * current_beta @ conditional_scatter
            + scatter[column, column]
        )
        gamma_draw = sample_gig(
            gamma_key,
            1.0 - n_observations / 2.0,
            jnp.maximum(gamma_chi, jnp.asarray(1e-12, dtype=dtype)),
            diagonal_rate,
        )
        gamma = jnp.where(
            gamma_draw.accepted,
            gamma_draw.value,
            jnp.asarray(1.0, dtype=dtype),
        )

        lane_outer = lane_mask[:, None] & lane_mask[None, :]
        conditional_block = conditional_precision[
            jnp.ix_(active_positions, active_positions)
        ]
        unmasked_beta_precision = (
            quadratic / gamma
            + jnp.diag(1.0 / current.tau[compact_indices, column])
            + diagonal_rate * conditional_block
        )
        beta_precision = jnp.where(lane_outer, unmasked_beta_precision, 0.0)
        beta_precision = beta_precision + jnp.diag((~lane_mask).astype(dtype))
        beta_precision = 0.5 * (beta_precision + beta_precision.T)
        beta_mean = jnp.linalg.solve(beta_precision, conditional_scatter) / gamma
        beta_cholesky = jnp.linalg.cholesky(beta_precision)
        dense_beta_noise = jax.random.normal(
            beta_key,
            (padded_count,),
            dtype=dtype,
        )
        beta_noise = jnp.linalg.solve(
            beta_cholesky.T,
            dense_beta_noise[active_positions],
        )
        compact_beta = jnp.where(lane_mask, beta_mean + beta_noise, 0.0)
        beta = (
            jnp.zeros((padded_count,), dtype=dtype)
            .at[active_positions]
            .add(compact_beta)
        )
        covariance, precision = update_covariance_column(
            current.covariance,
            current.precision,
            column_array,
            indices,
            beta,
            gamma,
        )

        phi_values, psi_values, tau_values, scales_accepted = (
            _update_sbm_local_scales(
                phi_key=phi_key,
                psi_key=psi_key,
                beta=compact_beta,
                active=lane_mask,
                active_count=jnp.sum(lane_mask, dtype=jnp.int32),
                random_positions=active_positions,
                padded_count=padded_count,
                current_phi=current.phi[compact_indices, column],
                current_psi=current.psi[compact_indices, column],
                current_tau=current.tau[compact_indices, column],
                a=a,
                b=b,
                tau1sq=tau1sq,
                dtype=dtype,
            )
        )
        active_lanes = (
            jnp.zeros((padded_count,), dtype=jnp.int32)
            .at[active_positions]
            .add(lane_mask.astype(jnp.int32))
            > 0
        )

        def expand(values: Array, fallback: Array) -> Array:
            updates = (
                jnp.zeros_like(fallback)
                .at[active_positions]
                .add(jnp.where(lane_mask, values, 0.0))
            )
            return jnp.where(active_lanes, updates, fallback)

        phi_values_dense = expand(phi_values, current.phi[indices, column])
        psi_values_dense = expand(psi_values, current.psi[indices, column])
        tau_values_dense = expand(tau_values, current.tau[indices, column])
        phi = current.phi.at[indices, column].set(phi_values_dense)
        phi = phi.at[column, indices].set(phi_values_dense)
        psi = current.psi.at[indices, column].set(psi_values_dense)
        psi = psi.at[column, indices].set(psi_values_dense)
        tau = current.tau.at[indices, column].set(tau_values_dense)
        tau = tau.at[column, indices].set(tau_values_dense)
        updated = BMState(covariance, precision, phi, psi, tau)
        finite = jnp.logical_and.reduce(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in updated])
        )
        accepted = sweep_accepted & gamma_draw.accepted & scales_accepted & finite
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


def prepare_sbm_compact_structure(
    active_mask: Array,
    other_indices: Array,
) -> SBMCompactStructure:
    """Build fixed-width active lane positions before compiled sampling."""
    mask = jnp.asarray(active_mask)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("active_mask must be a square two-dimensional array")
    dimension = mask.shape[0]
    mask = validate_sbm_active_mask(mask, dimension=dimension)

    indices = jnp.asarray(other_indices)
    if isinstance(indices, jax.core.Tracer):
        raise TypeError("prepare_sbm_compact_structure requires concrete other_indices")
    if indices.ndim != 2 or indices.shape != (dimension, dimension - 1):
        raise ValueError(
            f"other_indices must have shape ({dimension}, {dimension - 1})"
        )
    index_array = np.asarray(jax.device_get(indices))
    if not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError("other_indices must contain integer values")
    if np.any((index_array < 0) | (index_array >= dimension)):
        raise ValueError(
            f"other_indices values must be between zero and {dimension - 1}"
        )
    for column, row in enumerate(index_array):
        if column in row:
            raise ValueError("each other_indices row must exclude its column")
        if np.unique(row).size != dimension - 1:
            raise ValueError(
                "each other_indices row must contain every other index exactly once"
            )

    mask_array = np.asarray(jax.device_get(mask))
    column_positions = [
        np.flatnonzero(mask_array[index_array[column], column]).astype(np.int32)
        for column in range(dimension)
    ]
    compact_width = max((positions.size for positions in column_positions), default=0)
    active_positions = np.zeros((dimension, compact_width), dtype=np.int32)
    lane_mask = np.zeros((dimension, compact_width), dtype=np.bool_)
    for column, positions in enumerate(column_positions):
        active_positions[column, : positions.size] = positions
        lane_mask[column, : positions.size] = True
    target_device = next(iter(indices.devices()))
    return SBMCompactStructure(
        other_indices=indices,
        active_positions=jax.device_put(active_positions, target_device),
        lane_mask=jax.device_put(lane_mask, target_device),
    )
