import ast
import inspect
import textwrap
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from pybspcov.kernels import bm, sbm
from pybspcov.kernels.covariance import (
    _update_covariance_column_from_conditional_precision,
    update_covariance_column,
)


def test_precomputed_conditional_precision_update_matches_public_update() -> None:
    covariance = jnp.asarray(
        [[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.0]],
        dtype=jnp.float64,
    )
    precision = jnp.linalg.inv(covariance)
    column = jnp.asarray(1, dtype=jnp.int32)
    other_indices = jnp.asarray([0, 2], dtype=jnp.int32)
    beta = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    gamma = jnp.asarray(0.8, dtype=jnp.float64)
    block = precision[jnp.ix_(other_indices, other_indices)]
    cross = precision[other_indices, column]
    conditional_precision = block - jnp.outer(cross, cross) / precision[column, column]

    public_result = jax.jit(update_covariance_column)(
        covariance, precision, column, other_indices, beta, gamma
    )
    precomputed_result = jax.jit(_update_covariance_column_from_conditional_precision)(
        covariance,
        precision,
        column,
        other_indices,
        beta,
        gamma,
        conditional_precision,
    )

    for actual, expected in zip(precomputed_result, public_result, strict=True):
        assert jnp.array_equal(actual, expected)


def _nested_update_column(sweep: Callable[..., object]) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(sweep)))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update_column"
    )


@pytest.mark.parametrize(
    ("sweep", "expected_conditional_precision"),
    [
        (bm.bm_sweep, "moments.conditional_precision"),
        (sbm.sbm_sweep, "moments.conditional_precision"),
        (sbm.compact_sbm_sweep, "conditional_precision"),
    ],
    ids=["bm", "masked_dense_sbm", "compact_sbm"],
)
def test_production_sweeps_reuse_their_computed_conditional_precision(
    sweep: Callable[..., object],
    expected_conditional_precision: str,
) -> None:
    update_column = _nested_update_column(sweep)
    calls = [
        call
        for call in ast.walk(update_column)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]
    call_targets = {call.func.id for call in calls}
    helper_call = next(
        call
        for call in calls
        if call.func.id == "_update_covariance_column_from_conditional_precision"
    )

    assert "_update_covariance_column_from_conditional_precision" in call_targets
    assert "update_covariance_column" not in call_targets
    assert len(helper_call.args) == 7
    assert ast.unparse(helper_call.args[6]) == expected_conditional_precision
