"""Optional ArviZ, Matplotlib, and Seaborn visualizations."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import numpy.typing as npt

from pybspcov.model_selection import BandCVResult, ThresholdCVResult

type PlotType = Literal["heatmap", "uncertainty", "comparison"]


def _plotting() -> tuple[Any, Any]:
    try:
        pyplot = import_module("matplotlib.pyplot")
        seaborn = import_module("seaborn")
    except ModuleNotFoundError as error:
        if error.name not in {"matplotlib", "seaborn"}:
            raise
        raise ImportError(
            "plotting requires Matplotlib and Seaborn; install pybspcov[analysis]"
        ) from None
    return pyplot, seaborn


def _matrix(value: Any) -> npt.NDArray[np.floating[Any]]:
    if hasattr(value, "estimate"):
        value = value.estimate()
    matrix = np.asarray(jax.device_get(value))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("a square posterior covariance estimate is required")
    return matrix


def plot_trace(estimator: Any, *, row: int, column: int, **kwargs: Any) -> Any:
    """Plot one covariance entry across chains with ArviZ."""
    try:
        arviz = import_module("arviz")
    except ModuleNotFoundError as error:
        if error.name != "arviz":
            raise
        raise ImportError("plot_trace requires ArviZ; install pybspcov[analysis]") from None
    return arviz.plot_trace(
        estimator.to_arviz(),
        var_names=["covariance"],
        coords={"row": [row], "column": [column]},
        **kwargs,
    )


def plot_posterior_mean(
    estimator_or_matrix: Any,
    *,
    title: str = "Posterior mean covariance",
    color_limits: tuple[float, float] | None = None,
    color_low: str = "black",
    color_high: str = "white",
    show_values: bool = False,
    ax: Any = None,
) -> tuple[Any, Any]:
    """Plot a posterior mean covariance heatmap."""
    pyplot, seaborn = _plotting()
    matrix = _matrix(estimator_or_matrix)
    if ax is None:
        figure, ax = pyplot.subplots()
    else:
        figure = ax.figure
    cmap = seaborn.blend_palette([color_low, color_high], as_cmap=True)
    limits = {} if color_limits is None else {"vmin": color_limits[0], "vmax": color_limits[1]}
    seaborn.heatmap(matrix, ax=ax, cmap=cmap, annot=show_values, **limits)
    ax.set(title=title, xlabel="Variable", ylabel="Variable")
    return figure, ax


def plot_quantiles(
    estimator: Any,
    *,
    probs: npt.ArrayLike = (0.025, 0.5, 0.975),
    plot_type: PlotType = "heatmap",
    titles: list[str] | tuple[str, ...] | None = None,
    color_limits: tuple[float, float] | None = None,
) -> tuple[Any, list[Any]]:
    """Plot posterior covariance quantiles or interval widths."""
    pyplot, seaborn = _plotting()
    probabilities = np.atleast_1d(np.asarray(probs, dtype=float))
    quantiles = np.asarray(jax.device_get(estimator.quantile(probabilities)))
    if plot_type not in {"heatmap", "comparison", "uncertainty"}:
        raise ValueError("plot_type must be 'heatmap', 'comparison', or 'uncertainty'")
    if plot_type == "uncertainty":
        if quantiles.shape[0] != 2:
            raise ValueError("uncertainty plots require exactly two probabilities")
        matrices = [quantiles[1] - quantiles[0]]
        labels = ["Posterior interval width"]
    else:
        matrices = list(quantiles)
        labels = (
            list(titles)
            if titles is not None
            else [f"Quantile {probability:g}" for probability in probabilities]
        )
        if len(labels) != len(matrices):
            raise ValueError("titles must match the number of probabilities")
    figure, raw_axes = pyplot.subplots(
        1,
        len(matrices),
        squeeze=False,
        figsize=(5 * len(matrices), 4),
    )
    axes = list(raw_axes.ravel())
    limits = {} if color_limits is None else {"vmin": color_limits[0], "vmax": color_limits[1]}
    for matrix, label, axis in zip(matrices, labels, axes):
        seaborn.heatmap(matrix, ax=axis, **limits)
        axis.set(title=label, xlabel="Variable", ylabel="Variable")
    figure.tight_layout()
    return figure, axes


def save_quantile_plot(
    estimator: Any,
    filename: str | Path,
    **kwargs: Any,
) -> Path:
    """Save a quantile plot to an explicitly requested path."""
    destination = Path(filename)
    figure, _ = plot_quantiles(estimator, **kwargs)
    figure.savefig(destination, bbox_inches="tight")
    return destination


def plot_cv(
    result: BandCVResult | ThresholdCVResult,
    *,
    ax: Any = None,
) -> tuple[Any, Any]:
    """Plot one cross-validation score for every tuning combination."""
    pyplot, _ = _plotting()
    scores = np.asarray(jax.device_get(result.scores))
    if ax is None:
        figure, ax = pyplot.subplots()
    else:
        figure = ax.figure
    first, second, score = result.columns
    for value in np.unique(scores[:, 1]):
        subset = scores[scores[:, 1] == value]
        order = np.argsort(subset[:, 0])
        ax.plot(subset[order, 0], subset[order, 2], marker="o", label=f"{second}={value:g}")
    ax.set(xlabel=first, ylabel=score, title="Cross-validation")
    ax.legend()
    return figure, ax
