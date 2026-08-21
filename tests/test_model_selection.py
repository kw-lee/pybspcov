import jax
import jax.numpy as jnp

from pybspcov import (
    BandCVResult,
    ThresholdCVResult,
    cross_validate_band_ppp,
    cross_validate_threshold_ppp,
)


def _case() -> jax.Array:
    values = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    return values - values.mean(axis=0)


def test_band_cross_validation_returns_sorted_scores_and_best_parameters() -> None:
    result = cross_validate_band_ppp(
        _case(),
        bandwidths=[1, 2],
        epsilons=[0.0, 0.05],
        n_samples=2,
        key=jax.random.key(301),
        dtype="float32",
    )

    assert isinstance(result, BandCVResult)
    assert result.scores.shape == (4, 3)
    assert result.columns == ("bandwidth", "epsilon", "log_predictive_density")
    assert jnp.all(result.scores[:-1, 2] >= result.scores[1:, 2])
    assert result.best_bandwidth == int(result.scores[0, 0])
    assert result.best_epsilon == float(result.scores[0, 1])


def test_threshold_cross_validation_is_reproducible_and_sorted() -> None:
    arguments = {
        "thresholds": [0.0, 0.2],
        "epsilons": [0.0, 0.05],
        "n_samples": 2,
        "n_folds": 5,
        "dtype": "float32",
    }
    first = cross_validate_threshold_ppp(_case(), key=jax.random.key(307), **arguments)
    second = cross_validate_threshold_ppp(_case(), key=jax.random.key(307), **arguments)

    assert isinstance(first, ThresholdCVResult)
    assert first.columns == ("threshold", "epsilon", "spectral_norm_error")
    assert first.scores.shape == (4, 3)
    assert jnp.array_equal(first.scores, second.scores)
    assert jnp.all(first.scores[:-1, 2] <= first.scores[1:, 2])
    assert first.best_threshold == float(first.scores[0, 0])
    assert first.best_epsilon == float(first.scores[0, 1])


def test_cross_validation_requires_typed_scalar_key() -> None:
    try:
        cross_validate_band_ppp(
            _case(),
            bandwidths=[1],
            epsilons=[0.0],
            n_samples=1,
            key=jnp.asarray([1, 2], dtype=jnp.uint32),
            dtype="float32",
        )
    except TypeError as error:
        assert "typed JAX key" in str(error)
    else:
        raise AssertionError("legacy keys must be rejected")
