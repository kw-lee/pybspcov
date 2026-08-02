from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.extend import core as jax_core

from pybspcov import kernels
from pybspcov.kernels import sbm

ACTIVE_MASK = jnp.asarray(
    [
        [False, True, False, True],
        [True, False, True, False],
        [False, True, False, False],
        [True, False, False, False],
    ]
)
OTHER_INDICES = jnp.asarray(
    [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
    dtype=jnp.int32,
)


def _column_case(dtype: jnp.dtype) -> tuple[jax.Array, ...]:
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
            [0.9, 0.4, -0.3, 0.2],
            [-0.2, -0.1, 0.2, -0.3],
        ],
        dtype=dtype,
    )
    tau = jnp.asarray(
        [
            [0.4, 0.3, 0.2, 0.5],
            [0.3, 0.4, 0.6, 0.7],
            [0.2, 0.6, 0.4, 0.8],
            [0.5, 0.7, 0.8, 0.4],
        ],
        dtype=dtype,
    )
    return covariance, jnp.linalg.inv(covariance), x.T @ x, tau


def _compact_parameters(
    *,
    dtype: jnp.dtype,
    column: int,
    structure: sbm.SBMCompactStructure,
) -> sbm.SBMCompactColumnParameters:
    covariance, precision, scatter, tau = _column_case(dtype)
    return sbm.compact_sbm_column_parameters(
        covariance=covariance,
        precision=precision,
        scatter=scatter,
        tau=tau,
        column=jnp.asarray(column, dtype=jnp.int32),
        structure=structure,
        n_observations=jnp.asarray(6),
        diagonal_rate=jnp.asarray(1.0, dtype=dtype),
        gamma=jnp.asarray(0.9, dtype=dtype),
    )


def test_compact_sbm_public_api_exports_the_structure_and_conditionals() -> None:
    assert kernels.SBMCompactStructure is sbm.SBMCompactStructure
    assert kernels.SBMCompactColumnParameters is sbm.SBMCompactColumnParameters
    assert kernels.prepare_sbm_compact_structure is sbm.prepare_sbm_compact_structure
    assert kernels.compact_sbm_column_parameters is sbm.compact_sbm_column_parameters


def test_prepare_sbm_compact_structure_maps_literal_active_lanes() -> None:
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)

    assert jnp.array_equal(structure.other_indices, OTHER_INDICES)
    assert jnp.array_equal(
        structure.active_positions,
        jnp.asarray([[0, 2], [0, 1], [1, 0], [0, 0]], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        structure.lane_mask,
        jnp.asarray([[True, True], [True, True], [True, False], [True, False]]),
    )


def test_prepare_sbm_compact_structure_preserves_zero_width() -> None:
    structure = sbm.prepare_sbm_compact_structure(
        jnp.zeros((4, 4), dtype=jnp.bool_),
        OTHER_INDICES,
    )

    assert jnp.array_equal(structure.other_indices, OTHER_INDICES)
    assert structure.active_positions.shape == (4, 0)
    assert structure.active_positions.dtype == jnp.int32
    assert structure.lane_mask.shape == (4, 0)
    assert structure.lane_mask.dtype == jnp.bool_


@pytest.mark.parametrize(
    ("other_indices", "exception", "message"),
    [
        (jnp.zeros((4, 2), dtype=jnp.int32), ValueError, "shape"),
        (
            jnp.asarray(
                [[1, 1, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                dtype=jnp.int32,
            ),
            ValueError,
            "exactly once",
        ),
        (
            jnp.asarray(
                [[0, 1, 2], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                dtype=jnp.int32,
            ),
            ValueError,
            "exclude its column",
        ),
        (
            jnp.asarray(
                [[1, 2, 4], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                dtype=jnp.int32,
            ),
            ValueError,
            "between zero",
        ),
        (OTHER_INDICES.astype(jnp.float32), TypeError, "integer"),
    ],
)
def test_prepare_sbm_compact_structure_rejects_invalid_index_tables(
    other_indices: jax.Array,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        sbm.prepare_sbm_compact_structure(ACTIVE_MASK, other_indices)


def test_compact_sbm_structure_keeps_positions_bound_to_its_index_table() -> None:
    canonical_structure = sbm.prepare_sbm_compact_structure(
        ACTIVE_MASK,
        OTHER_INDICES,
    )
    reversed_structure = sbm.prepare_sbm_compact_structure(
        ACTIVE_MASK,
        OTHER_INDICES[:, ::-1],
    )

    canonical = _compact_parameters(
        dtype=jnp.float64,
        column=1,
        structure=canonical_structure,
    )
    reversed_result = _compact_parameters(
        dtype=jnp.float64,
        column=1,
        structure=reversed_structure,
    )

    assert jnp.allclose(canonical.gamma_chi, reversed_result.gamma_chi, rtol=1e-12)
    assert jnp.allclose(
        canonical.beta_mean,
        reversed_result.beta_mean[::-1],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_compact_sbm_conditionals_match_masked_dense_active_lanes(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    tolerance = 2e-5 if dtype_name == "float32" else 1e-11
    covariance, precision, scatter, tau = _column_case(dtype)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    column = jnp.asarray(2, dtype=jnp.int32)
    common = {
        "covariance": covariance,
        "precision": precision,
        "scatter": scatter,
        "tau": tau,
        "column": column,
        "n_observations": jnp.asarray(6),
        "diagonal_rate": jnp.asarray(1.0, dtype=dtype),
        "gamma": jnp.asarray(0.9, dtype=dtype),
    }

    masked = sbm.sbm_column_parameters(
        **common,
        active_mask=ACTIVE_MASK,
        other_indices=OTHER_INDICES[2],
    )
    compact = jax.jit(sbm.compact_sbm_column_parameters)(
        **common,
        structure=structure,
    )

    assert jnp.array_equal(compact.lane_mask, jnp.asarray([True, False]))
    assert compact.active_count == 1
    assert jnp.allclose(
        compact.conditional_precision,
        masked.conditional_precision,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.conditional_scatter[0],
        masked.conditional_scatter[1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.quadratic[0, 0],
        masked.quadratic[1, 1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.beta_precision[0, 0],
        masked.beta_precision[1, 1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.beta_mean[0],
        masked.beta_mean[1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.gamma_chi,
        masked.gamma_chi,
        rtol=tolerance,
        atol=tolerance,
    )
    assert compact.conditional_scatter[1] == 0.0
    assert compact.beta_mean[1] == 0.0
    assert jnp.array_equal(
        compact.beta_precision[1],
        jnp.asarray([0.0, 1.0], dtype=dtype),
    )
    assert jnp.array_equal(
        compact.beta_precision[:, 1],
        jnp.asarray([0.0, 1.0], dtype=dtype),
    )
    assert compact.beta_precision.dtype == dtype
    assert compact.beta_mean.dtype == dtype


def test_compact_sbm_conditionals_support_jit_of_vmap_over_columns() -> None:
    dtype = jnp.float32
    covariance, precision, scatter, tau = _column_case(dtype)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)

    def column_parameters(column: jax.Array) -> sbm.SBMCompactColumnParameters:
        return sbm.compact_sbm_column_parameters(
            covariance=covariance,
            precision=precision,
            scatter=scatter,
            tau=tau,
            column=column,
            structure=structure,
            n_observations=jnp.asarray(6),
            diagonal_rate=jnp.asarray(1.0, dtype=dtype),
            gamma=jnp.asarray(0.9, dtype=dtype),
        )

    parameters = jax.jit(jax.vmap(column_parameters))(jnp.arange(4, dtype=jnp.int32))

    assert parameters.conditional_scatter.shape == (4, 2)
    assert parameters.beta_precision.shape == (4, 2, 2)
    assert jnp.array_equal(parameters.lane_mask, structure.lane_mask)
    assert jnp.array_equal(parameters.active_count, jnp.asarray([2, 2, 1, 1]))


def test_compact_sbm_conditionals_preserve_zero_width() -> None:
    dtype = jnp.float32
    structure = sbm.prepare_sbm_compact_structure(
        jnp.zeros((4, 4), dtype=jnp.bool_),
        OTHER_INDICES,
    )

    parameters = jax.jit(
        lambda: _compact_parameters(dtype=dtype, column=1, structure=structure)
    )()

    assert parameters.active_count == 0
    assert parameters.conditional_scatter.shape == (0,)
    assert parameters.quadratic.shape == (0, 0)
    assert parameters.beta_precision.shape == (0, 0)
    assert parameters.beta_mean.shape == (0,)
    assert parameters.gamma_chi == _column_case(dtype)[2][1, 1]


def _nested_jaxprs(value: object) -> list[jax_core.Jaxpr]:
    found: list[jax_core.Jaxpr] = []
    if isinstance(value, jax_core.ClosedJaxpr):
        found.append(value.jaxpr)
    elif isinstance(value, jax_core.Jaxpr):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_nested_jaxprs(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.extend(_nested_jaxprs(item))
    return found


def test_compact_sbm_conditionals_factor_and_form_quadratic_at_compact_width() -> None:
    dtype = jnp.float32
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    closed = jax.make_jaxpr(
        lambda: _compact_parameters(dtype=dtype, column=1, structure=structure)
    )()
    pending = [closed.jaxpr]
    factor_shapes: list[tuple[int, ...]] = []
    matrix_products: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            if equation.primitive.name in {"cholesky", "lu", "triangular_solve"}:
                factor_shapes.extend(
                    variable.aval.shape
                    for variable in equation.invars
                    if hasattr(variable.aval, "shape") and len(variable.aval.shape) == 2
                )
            if equation.primitive.name == "dot_general":
                operand_shapes = [
                    variable.aval.shape
                    for variable in equation.invars[:2]
                    if hasattr(variable.aval, "shape")
                ]
                if len(operand_shapes) == 2 and all(
                    len(shape) == 2 for shape in operand_shapes
                ):
                    matrix_products.append((operand_shapes[0], operand_shapes[1]))
            for parameter in equation.params.values():
                pending.extend(_nested_jaxprs(parameter))

    assert (2, 2) in factor_shapes
    assert (3, 3) not in factor_shapes
    assert ((3, 3), (3, 3)) not in matrix_products
