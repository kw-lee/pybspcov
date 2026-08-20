"""Regression tests for Cholesky-based beta sampling in production sweeps."""

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core

from pybspcov.kernels import bm, sbm


def _nested_jaxprs(value: object) -> list[jax_core.Jaxpr]:
    if isinstance(value, jax_core.ClosedJaxpr):
        return [value.jaxpr]
    if isinstance(value, jax_core.Jaxpr):
        return [value]
    if isinstance(value, (tuple, list)):
        return [jaxpr for item in value for jaxpr in _nested_jaxprs(item)]
    if isinstance(value, dict):
        return [jaxpr for item in value.values() for jaxpr in _nested_jaxprs(item)]
    return []


def _primitive_counts(closed: jax_core.ClosedJaxpr) -> dict[str, int]:
    counts: dict[str, int] = {}
    pending = [closed.jaxpr]
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            name = equation.primitive.name
            counts[name] = counts.get(name, 0) + 1
            for parameter in equation.params.values():
                pending.extend(_nested_jaxprs(parameter))
    return counts


def _bm_sweep_jaxpr() -> jax_core.ClosedJaxpr:
    dtype = jnp.float32
    x = jnp.asarray(
        [
            [-1.0, 0.5, 0.2],
            [-0.4, -0.7, 0.1],
            [0.2, 0.1, -0.8],
            [0.5, -0.2, 0.6],
        ],
        dtype=dtype,
    )
    scatter = x.T @ x
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = bm.initialize_bm_state(
        jnp.diag(jnp.diag(scatter) / x.shape[0]),
        tau1sq,
    )
    return jax.make_jaxpr(bm.bm_sweep)(
        jax.random.key(17),
        state,
        scatter,
        jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32),
        jnp.asarray(x.shape[0]),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    )


def _compact_sbm_sweep_jaxpr() -> jax_core.ClosedJaxpr:
    dtype = jnp.float32
    covariance = jnp.asarray(
        [
            [2.0, 0.2, 0.0, 0.1],
            [0.2, 1.5, -0.15, 0.0],
            [0.0, -0.15, 1.2, 0.25],
            [0.1, 0.0, 0.25, 1.8],
        ],
        dtype=dtype,
    )
    x = jnp.asarray(
        [
            [-1.0, 0.5, 0.2, -0.1],
            [-0.4, -0.7, 0.1, 0.3],
            [0.2, 0.1, -0.8, 0.4],
            [0.5, -0.2, 0.6, -0.5],
        ],
        dtype=dtype,
    )
    active_mask = jnp.asarray(
        [
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
        ]
    )
    other_indices = jnp.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=jnp.int32,
    )
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, active_mask)
    structure = sbm.prepare_sbm_compact_structure(active_mask, other_indices)
    return jax.make_jaxpr(sbm.compact_sbm_sweep)(
        jax.random.key(19),
        state,
        x.T @ x,
        jnp.asarray(x.shape[0]),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        structure,
    )


def test_production_beta_sweeps_reuse_cholesky_for_mean_and_noise() -> None:
    """Catch a regression to independent LU solves for beta mean or noise."""
    for closed in (_bm_sweep_jaxpr(), _compact_sbm_sweep_jaxpr()):
        counts = _primitive_counts(closed)

        assert counts.get("cholesky", 0) == 1
        assert counts.get("triangular_solve", 0) >= 3
        assert counts.get("lu", 0) == 0
