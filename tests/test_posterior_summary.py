from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from pybspcov import BMSPCov, PosteriorSummary, SBMSPCov


def _fitted_estimator(factory: Callable[..., BMSPCov | SBMSPCov]) -> BMSPCov | SBMSPCov:
    model = factory(n_samples=2, burnin=0, n_chains=2, dtype="float32")
    model.posterior_samples_packed_ = jnp.asarray(
        [
            [[1.0, 10.0, 2.0], [3.0, 30.0, 4.0]],
            [[5.0, 50.0, 6.0], [7.0, 70.0, 8.0]],
        ],
        dtype=jnp.float32,
    )
    model.n_features_in_ = 2
    return model


@pytest.mark.parametrize("factory", [BMSPCov, SBMSPCov])
def test_estimate_combines_all_chains_and_samples(
    factory: Callable[..., BMSPCov | SBMSPCov],
) -> None:
    model = _fitted_estimator(factory)

    estimate = model.estimate()

    assert isinstance(estimate, jax.Array)
    assert estimate.dtype == jnp.float32
    assert estimate.devices() == model.posterior_samples_packed_.devices()
    assert jnp.array_equal(
        estimate,
        jnp.asarray([[4.0, 40.0], [40.0, 5.0]], dtype=jnp.float32),
    )


@pytest.mark.parametrize("factory", [BMSPCov, SBMSPCov])
def test_quantile_matches_r_type7_over_combined_chains(
    factory: Callable[..., BMSPCov | SBMSPCov],
) -> None:
    model = _fitted_estimator(factory)

    quantiles = model.quantile([0.25, 0.5, 0.75])

    assert quantiles.shape == (3, 2, 2)
    assert jnp.array_equal(
        quantiles,
        jnp.asarray(
            [
                [[2.5, 25.0], [25.0, 3.5]],
                [[4.0, 40.0], [40.0, 5.0]],
                [[5.5, 55.0], [55.0, 6.5]],
            ],
            dtype=jnp.float32,
        ),
    )


@pytest.mark.parametrize("factory", [BMSPCov, SBMSPCov])
def test_summary_reports_pooled_sample_statistics(
    factory: Callable[..., BMSPCov | SBMSPCov],
) -> None:
    model = _fitted_estimator(factory)

    summary = model.summary(probs=[0.25, 0.75])

    assert isinstance(summary, PosteriorSummary)
    assert jnp.array_equal(
        summary.mean,
        jnp.asarray([[4.0, 40.0], [40.0, 5.0]], dtype=jnp.float32),
    )
    assert jnp.allclose(
        summary.standard_deviation,
        jnp.asarray(
            [[2.5819888, 25.819889], [25.819889, 2.5819888]],
            dtype=jnp.float32,
        ),
        rtol=1e-6,
    )
    assert jnp.array_equal(summary.probabilities, jnp.asarray([0.25, 0.75]))
    assert jnp.array_equal(
        summary.quantiles,
        jnp.asarray(
            [
                [[2.5, 25.0], [25.0, 3.5]],
                [[5.5, 55.0], [55.0, 6.5]],
            ],
            dtype=jnp.float32,
        ),
    )
    assert summary.n_chains == 2
    assert summary.n_samples_per_chain == 2


@pytest.mark.parametrize("factory", [BMSPCov, SBMSPCov])
@pytest.mark.parametrize("method_name", ["estimate", "quantile", "summary"])
def test_posterior_summary_methods_require_a_fitted_estimator(
    factory: Callable[..., BMSPCov | SBMSPCov],
    method_name: str,
) -> None:
    model = factory(n_samples=1, burnin=0, dtype="float32")

    with pytest.raises(AttributeError, match="available only after fit"):
        getattr(model, method_name)()


@pytest.mark.parametrize("method_name", ["quantile", "summary"])
@pytest.mark.parametrize("probs", [[-0.1], [1.1], [float("nan")]])
def test_posterior_probabilities_must_be_finite_and_bounded(
    method_name: str,
    probs: list[float],
) -> None:
    model = _fitted_estimator(BMSPCov)

    with pytest.raises(ValueError, match="probabilities must be finite and between"):
        getattr(model, method_name)(probs=probs)


@pytest.mark.parametrize("method_name", ["quantile", "summary"])
def test_posterior_probabilities_must_not_be_empty(method_name: str) -> None:
    model = _fitted_estimator(BMSPCov)

    with pytest.raises(ValueError, match="probabilities must not be empty"):
        getattr(model, method_name)(probs=[])


@pytest.mark.parametrize("method_name", ["quantile", "summary"])
def test_posterior_probabilities_must_be_scalar_or_one_dimensional(
    method_name: str,
) -> None:
    model = _fitted_estimator(BMSPCov)

    with pytest.raises(
        ValueError,
        match="probabilities must be a scalar or one-dimensional",
    ):
        getattr(model, method_name)(probs=[[0.25, 0.75]])


@pytest.mark.parametrize("method_name", ["quantile", "summary"])
@pytest.mark.parametrize("probs", [["median"], [0.5 + 0.0j], [True]])
def test_posterior_probabilities_must_be_real_numbers(
    method_name: str,
    probs: object,
) -> None:
    model = _fitted_estimator(BMSPCov)

    with pytest.raises(TypeError, match="probabilities must contain real numbers"):
        getattr(model, method_name)(probs=probs)
