from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pybspcov import kernels
from pybspcov.kernels import screening
from pybspcov.kernels.screening import (
    estimate_fnr_cutoff,
    fnr_screening_mask,
    pairwise_jeffreys_bayes_factors,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_fixture(name: str) -> np.ndarray:
    return np.genfromtxt(
        FIXTURE_DIR / name,
        delimiter=",",
        missing_values="NA",
        filling_values=np.nan,
        dtype=np.float64,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_pairwise_jeffreys_scores_match_bayesfactor_fixture(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    x = jnp.asarray(_load_fixture("sbm_screening_x.csv"), dtype=dtype)
    expected = _load_fixture("sbm_screening_pairwise_bf.csv")
    lower = np.tril_indices(x.shape[1], k=-1)
    tolerance = 2e-6 if dtype_name == "float32" else 1e-12

    actual = pairwise_jeffreys_bayes_factors(x)
    compiled = jax.jit(screening._pairwise_jeffreys_bayes_factors_unchecked)(x)

    assert actual.dtype == dtype
    assert np.all(np.isnan(np.asarray(actual)[np.triu_indices(x.shape[1])]))
    np.testing.assert_allclose(
        np.asarray(actual)[lower],
        expected[lower],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(compiled),
        np.asarray(actual),
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (jnp.ones((4,)), "two-dimensional"),
        (jnp.ones((2, 2)), "at least three observations"),
        (jnp.ones((3, 1)), "at least two variables"),
        (jnp.asarray([[1.0, 2.0], [2.0, jnp.nan], [3.0, 4.0]]), "finite"),
        (jnp.asarray([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]), "constant"),
    ],
)
def test_pairwise_jeffreys_scores_validate_observations(
    x: jax.Array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pairwise_jeffreys_bayes_factors(x)


def test_pairwise_jeffreys_scores_reject_jit_tracers() -> None:
    x = jnp.arange(12.0).reshape(4, 3)

    with pytest.raises(TypeError, match="host validation.*cannot be used inside"):
        jax.jit(pairwise_jeffreys_bayes_factors)(x)


def test_pairwise_jeffreys_scores_promote_low_precision_inputs() -> None:
    x = jnp.arange(12, dtype=jnp.bfloat16).reshape(4, 3)

    scores = pairwise_jeffreys_bayes_factors(x)

    assert scores.dtype == jnp.float32


def test_pairwise_jeffreys_scores_match_upstream_for_perfect_correlation() -> None:
    x = jnp.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])

    scores = pairwise_jeffreys_bayes_factors(x)

    assert jnp.isinf(scores[1, 0])


def test_jeffreys_log_constant_is_stable_for_large_float32_samples() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("float64 reference requires JAX_ENABLE_X64=1")

    float32_score = screening._jeffreys_bayes_factor_from_correlation(
        jnp.asarray(0.0, dtype=jnp.float32),
        n_observations=10_000,
    )
    float64_score = screening._jeffreys_bayes_factor_from_correlation(
        jnp.asarray(0.0, dtype=jnp.float64),
        n_observations=10_000,
    )

    np.testing.assert_allclose(float32_score, float64_score, rtol=1e-5)


def test_fnr_public_functions_are_exported() -> None:
    assert kernels.estimate_fnr_cutoff is estimate_fnr_cutoff
    assert kernels.pairwise_jeffreys_bayes_factors is pairwise_jeffreys_bayes_factors


def test_fnr_screening_retains_an_infinite_bayes_factor() -> None:
    scores = jnp.asarray([[jnp.nan, jnp.nan], [jnp.inf, jnp.nan]])

    mask = fnr_screening_mask(scores, cutoff=1.0)

    np.testing.assert_array_equal(mask, [[False, True], [True, False]])


def test_fnr_screening_accepts_an_infinite_upstream_cutoff() -> None:
    cutoff = estimate_fnr_cutoff(
        jax.random.key(2707),
        n_observations=12,
        correlation=1.0,
        n_simulations=32,
        dtype="float32",
    )
    scores = jnp.asarray([[jnp.nan, jnp.nan], [jnp.inf, jnp.nan]])

    mask = fnr_screening_mask(scores, cutoff)

    np.testing.assert_array_equal(mask, jnp.zeros((2, 2), dtype=jnp.bool_))


def test_estimate_fnr_cutoff_matches_upstream_distribution() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")

    cutoff = estimate_fnr_cutoff(
        jax.random.key(314159),
        n_observations=12,
        correlation=0.25,
        false_negative_rate=0.05,
        n_simulations=8192,
        dtype="float64",
    )

    assert cutoff.dtype == jnp.float64
    assert float(cutoff) == pytest.approx(0.35605260903947278, abs=0.025)


def test_estimate_fnr_cutoff_uses_requested_correlation() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")

    cutoff = estimate_fnr_cutoff(
        jax.random.key(271828),
        n_observations=12,
        correlation=0.8,
        false_negative_rate=0.5,
        n_simulations=8192,
        dtype="float64",
    )

    # bspcov:::select_cutoff with seed=271828 and 8192 simulations gives
    # 36.230897813410905. The tolerance covers independent R/JAX RNG streams.
    assert float(cutoff) == pytest.approx(36.230897813410905, abs=6.0)


def test_estimate_fnr_cutoff_is_reproducible_in_float32() -> None:
    key = jax.random.key(2718)

    first = estimate_fnr_cutoff(
        key,
        n_observations=12,
        correlation=0.25,
        false_negative_rate=0.05,
        n_simulations=256,
        dtype="float32",
    )
    second = estimate_fnr_cutoff(
        key,
        n_observations=12,
        correlation=0.25,
        false_negative_rate=0.05,
        n_simulations=256,
        dtype="float32",
    )

    assert first.dtype == jnp.float32
    assert first.devices() == key.devices()
    assert jnp.array_equal(first, second)
    assert jnp.isfinite(first)
    assert first > 0.0


@pytest.mark.parametrize("correlation", [-1.0, 1.0])
@pytest.mark.parametrize(
    ("n_simulations", "false_negative_rate"),
    [(1, 0.0), (1, 0.05), (1, 1.0), (32, 0.05)],
)
def test_estimate_fnr_cutoff_accepts_upstream_perfect_correlation(
    correlation: float,
    n_simulations: int,
    false_negative_rate: float,
) -> None:
    cutoff = estimate_fnr_cutoff(
        jax.random.key(2729),
        n_observations=12,
        correlation=correlation,
        false_negative_rate=false_negative_rate,
        n_simulations=n_simulations,
        dtype="float32",
    )

    assert jnp.isinf(cutoff)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"n_observations": 2}, ValueError, "at least three"),
        ({"n_observations": True}, TypeError, "integer"),
        ({"n_simulations": 0}, ValueError, "positive"),
        ({"correlation": 1.01}, ValueError, "between -1 and 1"),
        ({"correlation": jnp.nan}, ValueError, "finite"),
        ({"false_negative_rate": -0.1}, ValueError, "between zero and one"),
        ({"false_negative_rate": jnp.nan}, ValueError, "finite"),
        ({"dtype": "bfloat16"}, ValueError, "float32.*float64"),
    ],
)
def test_estimate_fnr_cutoff_validates_configuration(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    defaults: dict[str, object] = {
        "n_observations": 12,
        "n_simulations": 32,
        "correlation": 0.25,
        "false_negative_rate": 0.05,
        "dtype": "float32",
    }
    defaults.update(kwargs)

    with pytest.raises(exception, match=message):
        estimate_fnr_cutoff(jax.random.key(1), **defaults)  # type: ignore[arg-type]


def test_estimate_fnr_cutoff_requires_a_typed_scalar_key() -> None:
    with pytest.raises(TypeError, match="typed JAX key"):
        estimate_fnr_cutoff(
            jax.random.PRNGKey(1),
            n_observations=12,
            n_simulations=32,
            dtype="float32",
        )
    with pytest.raises(ValueError, match="scalar key"):
        estimate_fnr_cutoff(
            jax.random.split(jax.random.key(1), 2),
            n_observations=12,
            n_simulations=32,
            dtype="float32",
        )
