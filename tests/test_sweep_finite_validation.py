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


def _whole_state_finite_predicates(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reduce"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "logical_and"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Attribute)
        and node.args[0].func.attr == "stack"
        and any(
            isinstance(argument, ast.ListComp) and bool(_isfinite_calls(argument))
            for argument in node.args[0].args
        )
    ]


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
    post_loop = ast.Module(
        body=function.body[function.body.index(column_loop) + 1 :],
        type_ignores=[],
    )
    assert len(_whole_state_finite_predicates(post_loop)) == 1
