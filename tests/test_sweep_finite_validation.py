"""Structural regressions for sweep-level finite-state validation."""

import ast
import inspect
from collections.abc import Callable

import pytest

from pybspcov.kernels.bm import bm_sweep
from pybspcov.kernels.sbm import compact_sbm_sweep, sbm_sweep


def _isfinite_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "isfinite"
    ]


def _has_updated_finite_check(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "finite"
            for target in statement.targets
        )
        and bool(_isfinite_calls(statement))
        and any(
            isinstance(node, ast.Name) and node.id == "updated"
            for node in ast.walk(statement.value)
        )
    )


@pytest.mark.parametrize("sweep", [bm_sweep, sbm_sweep, compact_sbm_sweep])
def test_sweep_validates_whole_state_after_column_loop(
    sweep: Callable[..., object],
) -> None:
    """A whole-state finite scan must run once per sweep, not per column."""
    function = ast.parse(inspect.getsource(sweep)).body[0]
    assert isinstance(function, ast.FunctionDef)
    update_column = next(
        statement
        for statement in function.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "update_column"
    )
    column_loop = next(
        statement
        for statement in function.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "fori_loop"
    )

    assert not _isfinite_calls(update_column)
    assert any(
        _has_updated_finite_check(statement)
        for statement in function.body[function.body.index(column_loop) + 1 :]
    )
