"""Cross-validation for post-processed posterior covariance estimators."""

from dataclasses import dataclass
from itertools import product
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from pybspcov.estimators import BandPPP, ThresholdPPP, _validate_key
from pybspcov.kernels.thresholdppp import ThresholdMethod

type DTypeName = Literal["float32", "float64"]


@dataclass(frozen=True)
class BandCVResult:
    """Sorted leave-one-out scores for BandPPP tuning parameters."""

    scores: Array
    columns: tuple[str, str, str]
    best_bandwidth: int
    best_epsilon: float


@dataclass(frozen=True)
class ThresholdCVResult:
    """Sorted fold errors for ThresholdPPP tuning parameters."""

    scores: Array
    columns: tuple[str, str, str]
    best_threshold: float
    best_epsilon: float


def _real_vector(name: str, values: ArrayLike, *, positive: bool) -> list[float]:
    try:
        array = jnp.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numbers") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if (
        not jnp.issubdtype(array.dtype, jnp.number)
        or jnp.issubdtype(array.dtype, jnp.complexfloating)
        or jnp.issubdtype(array.dtype, jnp.bool_)
    ):
        raise TypeError(f"{name} must contain real numbers")
    lower_ok = array > 0 if positive else array >= 0
    if not bool(jnp.all(jnp.isfinite(array) & lower_ok)):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain finite {qualifier} values")
    return [float(value) for value in array]


def _bandwidth_vector(values: ArrayLike, dimension: int) -> list[int]:
    numeric = _real_vector("bandwidths", values, positive=True)
    bandwidths = [int(value) for value in numeric]
    if any(float(integer) != value for integer, value in zip(bandwidths, numeric)):
        raise ValueError("bandwidths must contain integers")
    if any(value >= dimension for value in bandwidths):
        raise ValueError("bandwidths must be smaller than the feature dimension")
    return bandwidths


def _log_predictive_density(draws: Array, observation: Array) -> Array:
    sign, log_determinant = jnp.linalg.slogdet(draws)
    solved = jax.vmap(jnp.linalg.solve)(
        draws,
        jnp.broadcast_to(observation, (draws.shape[0], observation.shape[0])),
    )
    quadratic = jnp.einsum("i,si->s", observation, solved)
    dimension = observation.shape[0]
    log_density = -0.5 * (
        dimension * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic
    )
    log_density = jnp.where(sign > 0, log_density, -jnp.inf)
    return logsumexp(log_density) - jnp.log(draws.shape[0])


def cross_validate_band_ppp(
    X: ArrayLike,
    *,
    bandwidths: ArrayLike,
    epsilons: ArrayLike,
    key: Array,
    n_samples: int = 2000,
    prior_scale: ArrayLike | None = None,
    prior_df: float | Array | None = None,
    dtype: DTypeName = "float64",
    device: str | jax.Device | None = None,
) -> BandCVResult:
    """Evaluate BandPPP parameters with leave-one-out predictive density."""
    _validate_key(key)
    x = jnp.asarray(X, dtype=jnp.dtype(dtype))
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("X must have at least two rows and two columns")
    if not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError("X must contain only finite values")
    bandwidth_values = _bandwidth_vector(bandwidths, x.shape[1])
    epsilon_values = _real_vector("epsilons", epsilons, positive=False)
    combinations = list(product(bandwidth_values, epsilon_values))
    keys = jax.random.split(key, len(combinations) * x.shape[0])
    key_index = 0
    rows: list[tuple[float, float, float]] = []
    for bandwidth, epsilon in combinations:
        fold_scores: list[Array] = []
        for held_out in range(x.shape[0]):
            training = jnp.concatenate((x[:held_out], x[held_out + 1 :]), axis=0)
            model = BandPPP(
                bandwidth,
                epsilon=epsilon,
                prior_scale=prior_scale,
                prior_df=prior_df,
                n_samples=n_samples,
                dtype=dtype,
                device=device,
            ).fit(training, key=keys[key_index])
            key_index += 1
            draws = model.posterior_samples_.reshape((-1, x.shape[1], x.shape[1]))
            fold_scores.append(_log_predictive_density(draws, x[held_out]))
        rows.append((bandwidth, epsilon, float(jnp.mean(jnp.stack(fold_scores)))))
    scores = jnp.asarray(sorted(rows, key=lambda row: row[2], reverse=True))
    return BandCVResult(
        scores=scores,
        columns=("bandwidth", "epsilon", "log_predictive_density"),
        best_bandwidth=int(scores[0, 0]),
        best_epsilon=float(scores[0, 1]),
    )


def cross_validate_threshold_ppp(
    X: ArrayLike,
    *,
    thresholds: ArrayLike,
    epsilons: ArrayLike,
    key: Array,
    method: ThresholdMethod = "hard",
    n_samples: int = 2000,
    n_folds: int = 10,
    prior_scale: ArrayLike | None = None,
    prior_df: float | Array | None = None,
    dtype: DTypeName = "float64",
    device: str | jax.Device | None = None,
) -> ThresholdCVResult:
    """Evaluate threshold PPP parameters with spectral-norm fold error."""
    _validate_key(key)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int):
        raise TypeError("n_folds must be an integer")
    x = jnp.asarray(X, dtype=jnp.dtype(dtype))
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("X must have at least two rows and two columns")
    if not 2 <= n_folds <= x.shape[0]:
        raise ValueError("n_folds must be between 2 and the number of rows")
    threshold_values = _real_vector("thresholds", thresholds, positive=False)
    epsilon_values = _real_vector("epsilons", epsilons, positive=False)
    combinations = list(product(threshold_values, epsilon_values))
    all_keys = jax.random.split(key, 1 + len(combinations) * n_folds)
    permutation = jax.random.permutation(all_keys[0], x.shape[0])
    fold_labels = jnp.arange(x.shape[0]) % n_folds
    assignments = jnp.empty_like(fold_labels).at[permutation].set(fold_labels)
    key_index = 1
    rows: list[tuple[float, float, float]] = []
    for threshold, epsilon in combinations:
        fold_errors: list[Array] = []
        for fold in range(n_folds):
            validation = x[assignments == fold]
            training = x[assignments != fold]
            scale = prior_scale
            if scale is None:
                sample_covariance = jnp.cov(training, rowvar=False)
                scale = jnp.eye(x.shape[1], dtype=x.dtype) * jnp.mean(
                    jnp.diag(sample_covariance)
                )
            model = ThresholdPPP(
                threshold,
                method=method,
                epsilon=epsilon,
                prior_scale=scale,
                prior_df=prior_df,
                n_samples=n_samples,
                dtype=dtype,
                device=device,
            ).fit(training, key=all_keys[key_index])
            key_index += 1
            validation_covariance = validation.T @ validation / validation.shape[0]
            fold_errors.append(
                cast(Array, jnp.linalg.norm(model.estimate() - validation_covariance, 2))
            )
        rows.append((threshold, epsilon, float(jnp.mean(jnp.stack(fold_errors)))))
    scores = jnp.asarray(sorted(rows, key=lambda row: row[2]))
    return ThresholdCVResult(
        scores=scores,
        columns=("threshold", "epsilon", "spectral_norm_error"),
        best_threshold=float(scores[0, 0]),
        best_epsilon=float(scores[0, 1]),
    )
