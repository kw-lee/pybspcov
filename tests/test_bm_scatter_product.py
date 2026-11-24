"""Tests for the maintained ``precision @ scatter @ precision`` product."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from pybspcov.kernels import bm
from pybspcov.kernels.covariance import update_covariance_column


def _case(dtype: Any, dimension: int = 7) -> tuple[jax.Array, jax.Array, jax.Array]:
    factor_key, data_key = jax.random.split(jax.random.key(11))
    factor = jax.random.normal(factor_key, (dimension, dimension), dtype=dtype)
    covariance = factor @ factor.T / dimension + jnp.eye(dimension, dtype=dtype)
    x = jax.random.normal(data_key, (3 * dimension, dimension), dtype=dtype)
    return covariance, jnp.linalg.inv(covariance), x.T @ x


def _other_indices(dimension: int) -> jax.Array:
    return jnp.asarray(
        [[j for j in range(dimension) if j != column] for column in range(dimension)],
        dtype=jnp.int32,
    )


@pytest.mark.parametrize(
    ("dtype_name", "tolerance"),
    [("float32", 2e-4), ("float64", 1e-11)],
)
def test_column_moments_from_product_match_direct_moments(
    dtype_name: str,
    tolerance: float,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    covariance, precision, scatter = _case(dtype)
    other_indices = _other_indices(covariance.shape[0])
    product = precision @ scatter @ precision
    n_observations = jnp.asarray(21)
    diagonal_rate = jnp.asarray(1.0, dtype=dtype)

    for column in range(covariance.shape[0]):
        arguments = (
            jnp.asarray(column),
            other_indices[column],
            n_observations,
            diagonal_rate,
        )
        direct = bm._bm_column_moments(covariance, precision, scatter, *arguments)
        from_product, _ = bm._bm_column_moments_from_product(
            covariance, precision, scatter, product, *arguments
        )

        assert jnp.array_equal(
            from_product.conditional_precision, direct.conditional_precision
        )
        assert jnp.array_equal(
            from_product.conditional_scatter, direct.conditional_scatter
        )
        scale = jnp.max(jnp.abs(direct.quadratic))
        assert jnp.max(jnp.abs(from_product.quadratic - direct.quadratic)) <= (
            tolerance * scale
        )
        assert jnp.allclose(
            from_product.gamma_chi, direct.gamma_chi, rtol=tolerance * 10, atol=0.0
        )


@pytest.mark.parametrize(
    ("dtype_name", "tolerance"),
    [("float32", 2e-4), ("float64", 1e-11)],
)
def test_scatter_product_update_tracks_the_updated_precision(
    dtype_name: str,
    tolerance: float,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    covariance, precision, scatter = _case(dtype)
    dimension = covariance.shape[0]
    other_indices = _other_indices(dimension)
    product = precision @ scatter @ precision
    key = jax.random.key(23)

    for step in range(3 * dimension):
        column = jnp.asarray(step % dimension)
        indices = other_indices[column]
        key, beta_key, gamma_key = jax.random.split(key, 3)
        beta = covariance[indices, column] + 0.05 * jax.random.normal(
            beta_key, (dimension - 1,), dtype=dtype
        )
        gamma = 0.5 + jax.random.uniform(gamma_key, dtype=dtype)
        _, projected = bm._bm_column_moments_from_product(
            covariance,
            precision,
            scatter,
            product,
            column,
            indices,
            jnp.asarray(21),
            jnp.asarray(1.0, dtype=dtype),
        )
        covariance, updated_precision = update_covariance_column(
            covariance, precision, column, indices, beta, gamma
        )
        product = bm._update_scatter_product(
            projected, precision, updated_precision, scatter, column, gamma
        )
        precision = updated_precision

    expected = precision @ scatter @ precision
    assert jnp.max(jnp.abs(product - expected)) <= tolerance * jnp.max(
        jnp.abs(expected)
    )


def _matrix_products(jaxpr: Any) -> int:
    count = 0
    for equation in jaxpr.eqns:
        if equation.primitive.name == "dot_general" and all(
            variable.aval.ndim == 2 for variable in equation.invars
        ):
            count += 1
        for parameter in equation.params.values():
            candidates = parameter if isinstance(parameter, tuple) else (parameter,)
            for candidate in candidates:
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    count += _matrix_products(inner)
    return count


def test_bm_sweep_has_no_matrix_product_inside_the_column_loop() -> None:
    dtype = jnp.float32
    covariance, _, scatter = _case(dtype, dimension=5)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    jaxpr = jax.make_jaxpr(bm.bm_sweep)(
        jax.random.key(17),
        bm.initialize_bm_state(covariance, tau1sq),
        scatter,
        _other_indices(5),
        jnp.asarray(15),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    ).jaxpr

    top_level = sum(
        equation.primitive.name == "dot_general"
        and all(variable.aval.ndim == 2 for variable in equation.invars)
        for equation in jaxpr.eqns
    )
    # The product is rebuilt once per sweep; the column loop adds none.
    assert top_level == 2
    assert _matrix_products(jaxpr) == 2
