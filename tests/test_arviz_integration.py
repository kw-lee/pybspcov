from collections.abc import Callable

import arviz as az
import jax.numpy as jnp
import matplotlib
import pytest

from pybspcov import BMSPCov, SBMSPCov

matplotlib.use("Agg")


def _fitted_estimator(
    factory: Callable[..., BMSPCov | SBMSPCov],
) -> BMSPCov | SBMSPCov:
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


def _diagnostic_estimator() -> BMSPCov:
    model = BMSPCov(n_samples=20, burnin=0, n_chains=4, dtype="float32")
    draws = jnp.arange(80, dtype=jnp.float32).reshape(4, 20)
    model.posterior_samples_packed_ = jnp.stack(
        (draws + 1.0, draws / 10.0 - 2.0, draws + 2.0),
        axis=-1,
    )
    model.n_features_in_ = 2
    return model


@pytest.mark.parametrize("factory", [BMSPCov, SBMSPCov])
def test_to_arviz_preserves_chain_draw_and_covariance_axes(
    factory: Callable[..., BMSPCov | SBMSPCov],
) -> None:
    model = _fitted_estimator(factory)

    inference_data = model.to_arviz()

    covariance = inference_data.posterior["covariance"]
    assert covariance.dims == ("chain", "draw", "row", "column")
    assert covariance.shape == (2, 2, 2, 2)
    assert covariance.coords["chain"].values.tolist() == [0, 1]
    assert covariance.coords["draw"].values.tolist() == [0, 1]
    assert covariance.coords["row"].values.tolist() == [0, 1]
    assert covariance.coords["column"].values.tolist() == [0, 1]
    assert jnp.array_equal(
        covariance.values,
        jnp.asarray(
            [
                [
                    [[1.0, 10.0], [10.0, 2.0]],
                    [[3.0, 30.0], [30.0, 4.0]],
                ],
                [
                    [[5.0, 50.0], [50.0, 6.0]],
                    [[7.0, 70.0], [70.0, 8.0]],
                ],
            ],
            dtype=jnp.float32,
        ),
    )


def test_arviz_summary_reports_covariance_diagnostics() -> None:
    model = _diagnostic_estimator()

    summary = az.summary(
        model.to_arviz(),
        var_names=["covariance"],
        round_to="none",
    )

    assert {
        "mean",
        "sd",
        "ess_bulk",
        "ess_tail",
        "r_hat",
        "mcse_mean",
        "mcse_sd",
    }.issubset(summary.columns)
    assert summary.loc["covariance[0, 1]", "mean"] == pytest.approx(
        float(jnp.mean(model.posterior_samples_packed_[..., 1]))
    )


def test_to_arviz_uses_physical_chain_draw_order_under_custom_arviz_config() -> None:
    model = _fitted_estimator(BMSPCov)

    with az.rc_context({"data.sample_dims": ("draw", "chain")}):
        covariance = model.to_arviz().posterior["covariance"]

    assert covariance.dims == ("chain", "draw", "row", "column")
    assert covariance.shape == (2, 2, 2, 2)


def test_arviz_trace_plot_accepts_selected_covariance_entry() -> None:
    model = _diagnostic_estimator()

    plot = az.plot_trace(
        model.to_arviz(),
        var_names=["covariance"],
        coords={"row": [0], "column": [1]},
        backend="matplotlib",
    )

    assert plot.backend == "matplotlib"
    assert plot.viz["trace"]["covariance"].size == 4


@pytest.mark.parametrize("factory", [BMSPCov, SBMSPCov])
def test_to_arviz_requires_a_fitted_estimator(
    factory: Callable[..., BMSPCov | SBMSPCov],
) -> None:
    model = factory(n_samples=1, burnin=0, dtype="float32")

    with pytest.raises(AttributeError, match="to_arviz is available only after fit"):
        model.to_arviz()
