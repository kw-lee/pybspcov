import jax
import jax.numpy as jnp
import pytest

from pybspcov.kernels import screening
from pybspcov.kernels.screening import (
    correlation_screening_mask,
    fnr_screening_mask,
)


def test_fnr_screening_returns_active_edges_from_lower_triangle() -> None:
    scores = jnp.array(
        [
            [99.0, 99.0, 99.0, 99.0],
            [0.8, 99.0, 99.0, 99.0],
            [0.5, 1.2, 99.0, 99.0],
            [0.49, 0.7, 2.0, 99.0],
        ]
    )
    expected_active_mask = jnp.array(
        [
            [False, True, False, False],
            [True, False, True, True],
            [False, True, False, True],
            [False, True, True, False],
        ]
    )

    active_mask = fnr_screening_mask(scores, cutoff=0.5)

    assert jnp.array_equal(active_mask, expected_active_mask)
    assert active_mask.dtype == jnp.bool_


def test_public_fnr_screening_rejects_jit_tracers() -> None:
    scores = jnp.array(
        [[jnp.nan, jnp.nan, jnp.nan], [2.0, jnp.nan, jnp.nan], [0.1, 3.0, jnp.nan]]
    )

    with pytest.raises(
        TypeError, match="host validation.*cannot be used inside jax.jit"
    ):
        jax.jit(lambda values: fnr_screening_mask(values, cutoff=1.0))(scores)


def test_unchecked_fnr_kernel_has_a_static_jittable_shape() -> None:
    scores = jnp.array(
        [[jnp.nan, jnp.nan, jnp.nan], [2.0, jnp.nan, jnp.nan], [0.1, 3.0, jnp.nan]]
    )

    active_mask = jax.jit(screening._fnr_screening_mask_unchecked)(
        scores, jnp.asarray(1.0)
    )

    assert active_mask.shape == (3, 3)
    assert jnp.array_equal(active_mask, active_mask.T)
    assert not jnp.any(jnp.diag(active_mask))


@pytest.mark.parametrize(
    ("scores", "cutoff", "message"),
    [
        (jnp.ones((2, 3)), 1.0, "square"),
        (jnp.ones((1, 1)), 1.0, "at least two"),
        (jnp.array([[0.0, 0.0], [jnp.nan, 0.0]]), 1.0, "lower triangle"),
        (jnp.array([[0.0, 0.0], [-0.1, 0.0]]), 1.0, "non-negative"),
        (jnp.ones((2, 2)), -0.1, "non-negative"),
        (jnp.ones((2, 2)), jnp.inf, "finite"),
    ],
)
def test_fnr_screening_rejects_invalid_inputs(
    scores: jax.Array,
    cutoff: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fnr_screening_mask(scores, cutoff=cutoff)


def test_public_correlation_screening_rejects_jit_tracers() -> None:
    x = jnp.arange(12.0).reshape(4, 3)

    with pytest.raises(
        TypeError, match="host validation.*cannot be used inside jax.jit"
    ):
        jax.jit(
            lambda values: correlation_screening_mask(values, retained_fraction=0.5)
        )(x)


def test_unchecked_correlation_kernel_matches_hand_checked_case_under_jit() -> None:
    orthogonal_a = jnp.array([-1.0, -1.0, 1.0, 1.0])
    orthogonal_b = jnp.array([-1.0, 1.0, -1.0, 1.0])
    x = jnp.column_stack((orthogonal_a, orthogonal_a, orthogonal_b))
    expected_active_mask = jnp.array(
        [[False, True, False], [True, False, False], [False, False, False]]
    )

    active_mask = jax.jit(screening._correlation_screening_mask_unchecked)(
        x, jnp.asarray(0.5)
    )

    assert jnp.array_equal(active_mask, expected_active_mask)
    assert active_mask.dtype == jnp.bool_


def test_correlation_screening_excludes_ties_at_the_quantile() -> None:
    column = jnp.array([-1.0, -1.0, 1.0, 1.0])
    x = jnp.column_stack((column, column, column))

    active_mask = correlation_screening_mask(x, retained_fraction=0.5)

    assert not jnp.any(active_mask)


@pytest.mark.parametrize("retained_fraction", [0.0, 1.0])
def test_correlation_screening_accepts_probability_boundaries(
    retained_fraction: float,
) -> None:
    x = jnp.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

    active_mask = correlation_screening_mask(x, retained_fraction=retained_fraction)

    assert active_mask.shape == (2, 2)
    assert not jnp.any(active_mask)


@pytest.mark.parametrize(
    ("x", "retained_fraction", "message"),
    [
        (jnp.ones((4,)), 0.2, "two-dimensional"),
        (jnp.ones((1, 2)), 0.2, "at least two observations"),
        (jnp.ones((3, 1)), 0.2, "at least two variables"),
        (
            jnp.array([[0.0, 1.0], [1.0, jnp.nan], [2.0, 3.0]]),
            0.2,
            "finite",
        ),
        (jnp.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]), 0.2, "constant"),
        (jnp.arange(8.0).reshape(4, 2), -0.1, "between zero and one"),
        (jnp.arange(8.0).reshape(4, 2), 1.1, "between zero and one"),
        (jnp.arange(8.0).reshape(4, 2), jnp.nan, "finite"),
    ],
)
def test_correlation_screening_rejects_invalid_inputs(
    x: jax.Array,
    retained_fraction: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        correlation_screening_mask(x, retained_fraction=retained_fraction)
