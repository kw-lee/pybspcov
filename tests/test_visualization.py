from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

from pybspcov import BandCVResult, BandPPP
from pybspcov.visualization import (
    plot_cv,
    plot_posterior_mean,
    plot_quantiles,
    plot_trace,
    save_quantile_plot,
)


def _model() -> BandPPP:
    x = jnp.asarray([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=jnp.float32)
    return BandPPP(1, epsilon=0.01, n_samples=4, n_chains=2, dtype="float32").fit(
        x, key=jax.random.key(401)
    )


def test_posterior_mean_and_quantile_plots_return_figures() -> None:
    model = _model()

    mean_figure, mean_axis = plot_posterior_mean(model, show_values=True)
    quantile_figure, quantile_axes = plot_quantiles(model, probs=[0.1, 0.9])
    uncertainty_figure, uncertainty_axes = plot_quantiles(
        model, probs=[0.1, 0.9], plot_type="uncertainty"
    )

    assert mean_axis.get_title()
    assert len(quantile_axes) == 2
    assert len(uncertainty_axes) == 1
    mean_figure.canvas.draw()
    quantile_figure.canvas.draw()
    uncertainty_figure.canvas.draw()


def test_trace_and_cv_plots_accept_public_result_objects() -> None:
    model = _model()
    trace = plot_trace(model, row=0, column=1)
    result = BandCVResult(
        scores=jnp.asarray([[1.0, 0.0, -2.0], [2.0, 0.0, -3.0]]),
        columns=("bandwidth", "epsilon", "log_predictive_density"),
        best_bandwidth=1,
        best_epsilon=0.0,
    )
    figure, axis = plot_cv(result)
    method_figure, method_axis = result.plot()

    assert trace is not None
    assert axis.get_xlabel() == "bandwidth"
    figure.canvas.draw()
    method_figure.canvas.draw()
    assert method_axis.get_ylabel() == "log_predictive_density"


def test_save_quantile_plot_writes_only_requested_path(tmp_path: Path) -> None:
    destination = tmp_path / "quantiles.png"

    saved = save_quantile_plot(_model(), destination, probs=[0.1, 0.9])

    assert saved == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
