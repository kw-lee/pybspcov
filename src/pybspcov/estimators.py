"""Public estimators that orchestrate the pure JAX sampling kernels."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any, Literal, Self, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from pybspcov.kernels.bandppp import sample_bandppp_chains
from pybspcov.kernels.bm import (
    BMPackedChainResult,
    initialize_bm_state,
    sample_bm_packed_chains,
    unpack_lower_triangle_column_major,
)
from pybspcov.kernels.sbm import (
    SBMPackedChainResult,
    initialize_sbm_state,
    prepare_sbm_compact_structure,
    sample_compact_sbm_packed_chains,
    validate_sbm_active_mask,
)
from pybspcov.kernels.screening import (
    correlation_screening_mask,
    estimate_fnr_cutoff,
    fnr_screening_mask,
    pairwise_jeffreys_bayes_factors,
)
from pybspcov.kernels.thresholdppp import (
    ThresholdMethod,
    sample_thresholdppp_chains,
)

type _DTypeName = Literal["float32", "float64"]
type _DeviceName = Literal["cpu", "gpu", "cuda"]
type _DeviceRequest = _DeviceName | jax.Device | None
type _CutoffMethod = Literal["fnr", "correlation"]
type _BMRunner = Callable[..., BMPackedChainResult]
type _SBMRunner = Callable[..., SBMPackedChainResult]
type _BandPPPRunner = Callable[..., tuple[Array, Array]]
type _ThresholdPPPRunner = Callable[..., tuple[Array, Array]]


@dataclass(frozen=True)
class PosteriorSummary:
    """Elementwise posterior statistics pooled across fitted chains."""

    mean: Array
    standard_deviation: Array
    probabilities: Array
    quantiles: Array
    n_chains: int
    n_samples_per_chain: int


@dataclass(frozen=True)
class BMDiagnostics:
    """Execution status for a completed :class:`BMSPCov` fit."""

    accepted: Array
    n_sweeps: int
    n_rejected_sweeps: int
    n_initial_repairs: int
    initial_variance_floor: float
    dtype: str
    device: str


@dataclass(frozen=True)
class SBMDiagnostics:
    """Execution and screening status for a completed :class:`SBMSPCov` fit."""

    accepted: Array
    n_sweeps: int
    n_rejected_sweeps: int
    n_initial_repairs: int
    initial_variance_floor: float
    screening_jitter: float
    n_active_edges: int
    n_screened_edges: int
    cutoff_method: str
    dtype: str
    device: str


def _x64_enabled() -> bool:
    return bool(jax.config.x64_enabled)


def _available_devices(platform: str | None) -> list[jax.Device]:
    return jax.devices(platform)


@cache
def _compile_bm_chains() -> _BMRunner:
    return jax.jit(
        sample_bm_packed_chains,
        static_argnames=("burnin", "n_samples"),
    )


@cache
def _compile_sbm_chains() -> _SBMRunner:
    return jax.jit(
        sample_compact_sbm_packed_chains,
        static_argnames=("burnin", "n_samples"),
    )


@cache
def _compile_bandppp_chains() -> _BandPPPRunner:
    return jax.jit(
        sample_bandppp_chains,
        static_argnames=("n_samples",),
    )


@cache
def _compile_thresholdppp_chains() -> _ThresholdPPPRunner:
    return jax.jit(
        sample_thresholdppp_chains,
        static_argnames=("method", "n_samples"),
    )


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be positive")


def _validate_configuration(
    n_samples: int,
    burnin: int,
    n_chains: int,
    dtype: object,
    device: object,
) -> None:
    _validate_integer("n_samples", n_samples, minimum=1)
    _validate_integer("burnin", burnin, minimum=0)
    _validate_integer("n_chains", n_chains, minimum=1)
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if isinstance(device, str) and device not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be 'cpu', 'gpu', or 'cuda'")
    if device is not None and not isinstance(device, (str, jax.Device)):
        raise TypeError("device must be 'cpu', 'gpu', 'cuda', a JAX device, or None")


def _validate_bounded_scalar(
    name: str,
    value: float | Array,
    *,
    lower: float,
    upper: float,
) -> None:
    try:
        scalar = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not jnp.issubdtype(scalar.dtype, jnp.number) or jnp.issubdtype(
        scalar.dtype,
        jnp.complexfloating,
    ):
        raise TypeError(f"{name} must be a real scalar")
    if not bool(jnp.isfinite(scalar)) or not bool(
        (scalar >= lower) & (scalar <= upper)
    ):
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")


def _validate_optional_epsilon(value: object) -> None:
    if value is None:
        return
    try:
        scalar = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("epsilon must be a real scalar") from error
    if scalar.ndim != 0:
        raise ValueError("epsilon must be a scalar")
    if (
        not jnp.issubdtype(scalar.dtype, jnp.number)
        or jnp.issubdtype(scalar.dtype, jnp.complexfloating)
        or jnp.issubdtype(scalar.dtype, jnp.bool_)
    ):
        raise TypeError("epsilon must be a real scalar")
    if not bool(jnp.isfinite(scalar)) or not bool(scalar >= 0.0):
        raise ValueError("epsilon must be finite and non-negative")


def _validate_sbm_configuration(
    n_samples: int,
    burnin: int,
    n_chains: int,
    cutoff_method: object,
    fnr_correlation: float | Array,
    false_negative_rate: float | Array,
    n_cutoff_simulations: int,
    retained_fraction: float | Array,
    dtype: object,
    device: object,
) -> None:
    _validate_configuration(n_samples, burnin, n_chains, dtype, device)
    _validate_integer("n_cutoff_simulations", n_cutoff_simulations, minimum=1)
    if cutoff_method not in {"fnr", "correlation"}:
        raise ValueError("cutoff_method must be 'fnr' or 'correlation'")
    _validate_bounded_scalar(
        "fnr_correlation",
        fnr_correlation,
        lower=-1.0,
        upper=1.0,
    )
    _validate_bounded_scalar(
        "false_negative_rate",
        false_negative_rate,
        lower=0.0,
        upper=1.0,
    )
    _validate_bounded_scalar(
        "retained_fraction",
        retained_fraction,
        lower=0.0,
        upper=1.0,
    )


def _resolve_device(request: _DeviceRequest) -> jax.Device:
    if isinstance(request, jax.Device):
        return request
    platform = (
        None if request is None else "cuda" if request in {"gpu", "cuda"} else "cpu"
    )
    try:
        devices = _available_devices(platform)
    except (AssertionError, RuntimeError, ValueError) as error:
        if platform == "cuda":
            raise RuntimeError(
                "A CUDA device was explicitly requested, but the CUDA backend "
                "is unavailable. Install a compatible JAX CUDA build and verify "
                "jax.devices('cuda')."
            ) from error
        raise RuntimeError("The requested CPU backend is unavailable") from error
    if not devices:
        if platform == "cuda":
            raise RuntimeError(
                "A CUDA device was explicitly requested, but no CUDA device "
                "is available."
            )
        raise RuntimeError("No CPU device is available")
    return devices[0]


def _validate_key(key: Array) -> None:
    if not isinstance(key, jax.Array) or not jnp.issubdtype(
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError("key must be a typed JAX key from jax.random.key")
    if key.ndim != 0:
        raise ValueError("key must be a scalar key; pass one unsplit master key")


def _other_indices(dimension: int, device: jax.Device) -> Array:
    rows = [
        [index for index in range(dimension) if index != column]
        for column in range(dimension)
    ]
    return cast(Array, jax.device_put(jnp.asarray(rows, dtype=jnp.int32), device))


def _validate_x(x: Array) -> tuple[int, int]:
    if x.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    n_observations, dimension = x.shape
    if n_observations < 2:
        raise ValueError("X must contain at least two rows")
    if dimension < 2:
        raise ValueError("X must contain at least two columns")
    if not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError("X must contain only finite values")
    return n_observations, dimension


def _default_initial_covariance(x: Array) -> tuple[Array, int, float]:
    variances = jnp.var(x, axis=0, ddof=1)
    scale = jnp.maximum(jnp.max(jnp.abs(variances)), jnp.asarray(1.0, x.dtype))
    epsilon = 1.1920928955078125e-7 if x.dtype == jnp.float32 else 2.220446049250313e-16
    floor = jnp.asarray(epsilon, x.dtype) * scale
    repairs = int(jnp.count_nonzero(variances < floor))
    return jnp.diag(jnp.maximum(variances, floor)), repairs, float(floor)


class _PosteriorSummariesMixin:
    posterior_samples_packed_: Array
    n_features_in_: int

    def _require_posterior_samples(self, method_name: str) -> Array:
        try:
            return self.posterior_samples_packed_
        except AttributeError:
            raise AttributeError(f"{method_name} is available only after fit") from None

    @staticmethod
    def _probabilities(probs: ArrayLike | Sequence[float], packed: Array) -> Array:
        try:
            values = jnp.asarray(probs)
        except (TypeError, ValueError) as error:
            raise TypeError("probabilities must contain real numbers") from error
        if (
            not jnp.issubdtype(values.dtype, jnp.number)
            or jnp.issubdtype(values.dtype, jnp.complexfloating)
            or jnp.issubdtype(values.dtype, jnp.bool_)
        ):
            raise TypeError("probabilities must contain real numbers")
        probabilities = cast(
            Array,
            jax.device_put(
                jnp.atleast_1d(values),
                next(iter(packed.devices())),
            ),
        )
        if probabilities.ndim > 1:
            raise ValueError("probabilities must be a scalar or one-dimensional array")
        if probabilities.size == 0:
            raise ValueError("probabilities must not be empty")
        if not bool(
            jnp.all(
                jnp.isfinite(probabilities)
                & (probabilities >= 0.0)
                & (probabilities <= 1.0)
            )
        ):
            raise ValueError("probabilities must be finite and between 0 and 1")
        return probabilities

    def estimate(self) -> Array:
        """Return the posterior mean covariance pooled across all chains."""
        packed = self._require_posterior_samples("estimate")
        return unpack_lower_triangle_column_major(
            jnp.mean(packed, axis=(0, 1)),
            dimension=self.n_features_in_,
        )

    def quantile(
        self,
        probs: ArrayLike | Sequence[float] = (0.025, 0.5, 0.975),
    ) -> Array:
        """Return elementwise posterior covariance quantiles."""
        packed = self._require_posterior_samples("quantile")
        probabilities = self._probabilities(probs, packed)
        packed_quantiles = jnp.quantile(
            packed,
            probabilities,
            axis=(0, 1),
            method="linear",
        )
        return unpack_lower_triangle_column_major(
            packed_quantiles,
            dimension=self.n_features_in_,
        )

    def summary(
        self,
        probs: ArrayLike | Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
    ) -> PosteriorSummary:
        """Return elementwise posterior statistics pooled across all chains."""
        packed = self._require_posterior_samples("summary")
        probabilities = self._probabilities(probs, packed)
        standard_deviation = unpack_lower_triangle_column_major(
            jnp.std(packed, axis=(0, 1), ddof=1),
            dimension=self.n_features_in_,
        )
        return PosteriorSummary(
            mean=self.estimate(),
            standard_deviation=standard_deviation,
            probabilities=probabilities,
            quantiles=self.quantile(probabilities),
            n_chains=packed.shape[0],
            n_samples_per_chain=packed.shape[1],
        )

    def to_arviz(self) -> Any:
        """Return retained covariance draws as an ArviZ DataTree.

        ArviZ is an optional dependency. Install pybspcov[analysis] before
        calling this method. Conversion transfers fitted JAX arrays to host
        memory while preserving their chain and draw axes.
        """
        packed = self._require_posterior_samples("to_arviz")
        try:
            arviz = import_module("arviz")
        except ModuleNotFoundError as error:
            if error.name != "arviz":
                raise
            raise ImportError(
                "to_arviz requires ArviZ; install pybspcov[analysis]"
            ) from None

        covariance = unpack_lower_triangle_column_major(
            packed,
            dimension=self.n_features_in_,
        )
        features = range(self.n_features_in_)
        return arviz.from_dict(
            {"posterior": {"covariance": jax.device_get(covariance)}},
            sample_dims=["chain", "draw"],
            dims={"covariance": ["row", "column"]},
            coords={"row": features, "column": features},
        )


class BandPPP(_PosteriorSummariesMixin):
    """Post-processed posterior estimator for a banded covariance matrix.

    ``X`` follows upstream ``bspcov::bandPPP``: rows are observations,
    columns are variables, and the posterior scale uses ``X.T @ X`` without
    silently centering the data. Each chain consists of independent
    inverse-Wishart draws, followed by banding and an eigenvalue-floor
    adjustment; no burn-in is required.

    Args:
        bandwidth: R's ``k`` parameter. Entries more than this many diagonals
            from the main diagonal are set to zero.
        epsilon: R's ``eps`` eigenvalue floor. The upstream default is used
            when omitted.
        prior_scale: R's inverse-Wishart scale ``A``. Defaults to identity.
        prior_df: R's inverse-Wishart degrees of freedom ``nu``. Defaults to
            ``p + bandwidth`` after the feature dimension is known.
        n_samples: Number of retained posterior draws per chain.
        n_chains: Number of independent posterior chains.
        dtype: JAX floating-point dtype.
        device: JAX device or CPU/GPU platform request.
    """

    covariance_: Array
    posterior_samples_packed_: Array
    adjusted_draws_: Array
    epsilon_: Array
    prior_scale_: Array
    prior_df_: Array
    posterior_scale_: Array
    posterior_df_: Array
    n_features_in_: int
    n_observations_: int
    dtype_: jnp.dtype
    device_: jax.Device

    def __init__(
        self,
        bandwidth: int,
        *,
        epsilon: float | Array | None = None,
        prior_scale: ArrayLike | None = None,
        prior_df: float | Array | None = None,
        n_samples: int = 2000,
        n_chains: int = 1,
        dtype: _DTypeName = "float64",
        device: _DeviceRequest = None,
    ) -> None:
        _validate_integer("bandwidth", bandwidth, minimum=1)
        _validate_configuration(n_samples, 0, n_chains, dtype, device)
        _validate_optional_epsilon(epsilon)
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        self.prior_scale = prior_scale
        self.prior_df = prior_df
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.dtype = dtype
        self.device = device

    @property
    def posterior_samples_(self) -> Array:
        """Reconstruct full covariance draws from packed fitted storage."""
        try:
            packed = self.posterior_samples_packed_
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError(
                "posterior_samples_ is available only after fit"
            ) from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    def fit(self, X: ArrayLike, *, key: Array) -> Self:
        """Draw the inverse-Wishart posterior and apply BandPPP processing."""
        _validate_key(key)
        _validate_integer("bandwidth", self.bandwidth, minimum=1)
        _validate_configuration(
            self.n_samples,
            0,
            self.n_chains,
            self.dtype,
            self.device,
        )
        _validate_optional_epsilon(self.epsilon)
        if self.dtype == "float64" and not _x64_enabled():
            raise RuntimeError(
                "dtype='float64' requires JAX X64 mode. Set JAX_ENABLE_X64=1 "
                "before starting Python."
            )
        target = _resolve_device(self.device)
        dtype = jnp.dtype(self.dtype)

        with jax.default_device(target):
            try:
                raw_x = jnp.asarray(X)
            except (TypeError, ValueError) as error:
                raise TypeError("X must be a real numeric array") from error
            if not jnp.issubdtype(raw_x.dtype, jnp.number) or jnp.issubdtype(
                raw_x.dtype,
                jnp.complexfloating,
            ):
                raise TypeError("X must be a real numeric array")
            x = cast(Array, jax.device_put(jnp.asarray(raw_x, dtype=dtype), target))
            n_observations, dimension = _validate_x(x)

            if self.prior_scale is None:
                prior_scale = jnp.eye(dimension, dtype=dtype)
            else:
                try:
                    raw_prior_scale = jnp.asarray(self.prior_scale)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "prior_scale must be a real numeric array"
                    ) from error
                if (
                    not jnp.issubdtype(raw_prior_scale.dtype, jnp.number)
                    or jnp.issubdtype(raw_prior_scale.dtype, jnp.complexfloating)
                    or jnp.issubdtype(raw_prior_scale.dtype, jnp.bool_)
                ):
                    raise TypeError("prior_scale must be a real numeric array")
                prior_scale = jnp.asarray(raw_prior_scale, dtype=dtype)
            _validate_bandppp_prior_scale(prior_scale, dimension)
            prior_df = _bandppp_prior_df(
                self.prior_df,
                default=dimension + self.bandwidth,
                dimension=dimension,
                dtype=dtype,
            )
            posterior_df = prior_df + n_observations
            posterior_scale = x.T @ x + prior_scale
            epsilon = jnp.asarray(
                (jnp.log(self.bandwidth) ** 2)
                * (self.bandwidth + jnp.log(dimension))
                / n_observations
                if self.epsilon is None
                else self.epsilon,
                dtype=dtype,
            )
            chain_keys = jax.random.split(jax.device_put(key, target), self.n_chains)
            packed, adjusted = _compile_bandppp_chains()(
                chain_keys,
                posterior_scale,
                posterior_df,
                jnp.asarray(self.bandwidth, dtype=jnp.int32),
                epsilon,
                n_samples=self.n_samples,
            )
            packed.block_until_ready()

        self.posterior_samples_packed_ = packed
        self.covariance_ = unpack_lower_triangle_column_major(
            jnp.mean(packed, axis=(0, 1)),
            dimension=dimension,
        )
        self.adjusted_draws_ = adjusted
        self.epsilon_ = epsilon
        self.prior_scale_ = prior_scale
        self.prior_df_ = prior_df
        self.posterior_scale_ = posterior_scale
        self.posterior_df_ = posterior_df
        self.n_features_in_ = dimension
        self.n_observations_ = n_observations
        self.dtype_ = dtype
        self.device_ = target
        return self


class ThresholdPPP(_PosteriorSummariesMixin):
    """Post-processed posterior estimator for a sparse covariance matrix."""

    covariance_: Array
    posterior_samples_packed_: Array
    adjusted_draws_: Array
    epsilon_: Array
    threshold_: Array
    prior_scale_: Array
    prior_df_: Array
    posterior_scale_: Array
    posterior_df_: Array
    n_features_in_: int
    n_observations_: int
    dtype_: jnp.dtype
    device_: jax.Device

    def __init__(
        self,
        threshold: float | Array = 0.1,
        *,
        method: ThresholdMethod = "hard",
        epsilon: float | Array = 0.0,
        prior_scale: ArrayLike | None = None,
        prior_df: float | Array | None = None,
        n_samples: int = 2000,
        n_chains: int = 1,
        dtype: _DTypeName = "float64",
        device: _DeviceRequest = None,
    ) -> None:
        _validate_nonnegative_scalar("threshold", threshold)
        _validate_threshold_method(method)
        _validate_optional_epsilon(epsilon)
        _validate_configuration(n_samples, 0, n_chains, dtype, device)
        self.threshold = threshold
        self.method = method
        self.epsilon = epsilon
        self.prior_scale = prior_scale
        self.prior_df = prior_df
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.dtype = dtype
        self.device = device

    @property
    def posterior_samples_(self) -> Array:
        """Reconstruct full covariance draws from packed fitted storage."""
        try:
            packed = self.posterior_samples_packed_
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError(
                "posterior_samples_ is available only after fit"
            ) from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    def fit(self, X: ArrayLike, *, key: Array) -> Self:
        """Draw inverse-Wishart samples and apply sparse post-processing."""
        _validate_key(key)
        _validate_nonnegative_scalar("threshold", self.threshold)
        _validate_threshold_method(self.method)
        _validate_optional_epsilon(self.epsilon)
        _validate_configuration(
            self.n_samples, 0, self.n_chains, self.dtype, self.device
        )
        if self.dtype == "float64" and not _x64_enabled():
            raise RuntimeError(
                "dtype='float64' requires JAX X64 mode. Set JAX_ENABLE_X64=1 "
                "before starting Python."
            )
        target = _resolve_device(self.device)
        dtype = jnp.dtype(self.dtype)

        with jax.default_device(target):
            try:
                raw_x = jnp.asarray(X)
            except (TypeError, ValueError) as error:
                raise TypeError("X must be a real numeric array") from error
            if not jnp.issubdtype(raw_x.dtype, jnp.number) or jnp.issubdtype(
                raw_x.dtype, jnp.complexfloating
            ):
                raise TypeError("X must be a real numeric array")
            x = cast(Array, jax.device_put(jnp.asarray(raw_x, dtype=dtype), target))
            n_observations, dimension = _validate_x(x)

            if self.prior_scale is None:
                prior_scale = jnp.eye(dimension, dtype=dtype)
            else:
                try:
                    raw_prior_scale = jnp.asarray(self.prior_scale)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "prior_scale must be a real numeric array"
                    ) from error
                if (
                    not jnp.issubdtype(raw_prior_scale.dtype, jnp.number)
                    or jnp.issubdtype(raw_prior_scale.dtype, jnp.complexfloating)
                    or jnp.issubdtype(raw_prior_scale.dtype, jnp.bool_)
                ):
                    raise TypeError("prior_scale must be a real numeric array")
                prior_scale = jnp.asarray(raw_prior_scale, dtype=dtype)
            _validate_bandppp_prior_scale(prior_scale, dimension)
            prior_df = _bandppp_prior_df(
                self.prior_df,
                default=dimension + 1,
                dimension=dimension,
                dtype=dtype,
            )
            posterior_df = prior_df + n_observations
            posterior_scale = x.T @ x + prior_scale
            threshold = jnp.asarray(self.threshold, dtype=dtype)
            epsilon = jnp.asarray(self.epsilon, dtype=dtype)
            chain_keys = jax.random.split(jax.device_put(key, target), self.n_chains)
            packed, adjusted = _compile_thresholdppp_chains()(
                chain_keys,
                posterior_scale,
                posterior_df,
                threshold,
                epsilon,
                method=self.method,
                n_samples=self.n_samples,
            )
            packed.block_until_ready()

        self.posterior_samples_packed_ = packed
        self.covariance_ = unpack_lower_triangle_column_major(
            jnp.mean(packed, axis=(0, 1)), dimension=dimension
        )
        self.adjusted_draws_ = adjusted
        self.threshold_ = threshold
        self.epsilon_ = epsilon
        self.prior_scale_ = prior_scale
        self.prior_df_ = prior_df
        self.posterior_scale_ = posterior_scale
        self.posterior_df_ = posterior_df
        self.n_features_in_ = dimension
        self.n_observations_ = n_observations
        self.dtype_ = dtype
        self.device_ = target
        return self


def _validate_initial_covariance(covariance: Array, dimension: int) -> None:
    if covariance.shape != (dimension, dimension):
        raise ValueError(
            "initial_covariance must have shape "
            f"({dimension}, {dimension}); received {covariance.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(covariance))):
        raise ValueError("initial_covariance must contain only finite values")
    tolerance = 1e-5 if covariance.dtype == jnp.float32 else 1e-12
    if not bool(
        jnp.allclose(
            covariance,
            covariance.T,
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        raise ValueError("initial_covariance must be symmetric")
    if not bool(jnp.all(jnp.linalg.eigvalsh(covariance) > 0.0)):
        raise ValueError("initial_covariance must be positive definite")


def _validate_bandppp_prior_scale(scale: Array, dimension: int) -> None:
    if scale.shape != (dimension, dimension):
        raise ValueError(
            "prior_scale must have shape "
            f"({dimension}, {dimension}); received {scale.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(scale))):
        raise ValueError("prior_scale must contain only finite values")
    tolerance = 1e-5 if scale.dtype == jnp.float32 else 1e-12
    if not bool(jnp.allclose(scale, scale.T, rtol=tolerance, atol=tolerance)):
        raise ValueError("prior_scale must be symmetric")
    if not bool(jnp.all(jnp.linalg.eigvalsh(scale) > 0.0)):
        raise ValueError("prior_scale must be positive definite")


def _validate_nonnegative_scalar(name: str, value: object) -> None:
    try:
        scalar = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if (
        not jnp.issubdtype(scalar.dtype, jnp.number)
        or jnp.issubdtype(scalar.dtype, jnp.complexfloating)
        or jnp.issubdtype(scalar.dtype, jnp.bool_)
    ):
        raise TypeError(f"{name} must be a real scalar")
    if not bool(jnp.isfinite(scalar)) or not bool(scalar >= 0.0):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_threshold_method(method: object) -> None:
    if not isinstance(method, str):
        raise TypeError("method must be 'hard' or 'soft'")
    if method not in {"hard", "soft"}:
        raise ValueError("method must be 'hard' or 'soft'")


def _bandppp_prior_df(
    value: object,
    *,
    default: int,
    dimension: int,
    dtype: jnp.dtype,
) -> Array:
    try:
        raw = jnp.asarray(default if value is None else value)
    except (TypeError, ValueError) as error:
        raise TypeError("prior_df must be a real scalar") from error
    if raw.ndim != 0:
        raise ValueError("prior_df must be a scalar")
    if (
        not jnp.issubdtype(raw.dtype, jnp.number)
        or jnp.issubdtype(raw.dtype, jnp.complexfloating)
        or jnp.issubdtype(raw.dtype, jnp.bool_)
    ):
        raise TypeError("prior_df must be a real scalar")
    scalar = jnp.asarray(raw, dtype=dtype)
    if not bool(jnp.isfinite(scalar)):
        raise ValueError("prior_df must be finite")
    if not bool(scalar > dimension - 1):
        raise ValueError(f"prior_df must be greater than p - 1 ({dimension - 1})")
    return scalar


class BMSPCov(_PosteriorSummariesMixin):
    """Beta-mixture sparse covariance estimator.

    ``X`` is interpreted like upstream ``bspcov::bmspcov``: rows are
    observations, columns are variables, and the sampler uses ``X.T @ X``
    without silently centering the data. Center data before calling
    :meth:`fit` when the mean is unknown.
    """

    covariance_: Array
    posterior_samples_packed_: Array
    phi_samples_packed_: Array
    diagnostics_: BMDiagnostics
    initial_covariance_: Array
    n_features_in_: int
    n_observations_: int
    dtype_: jnp.dtype
    device_: jax.Device

    def __init__(
        self,
        *,
        n_samples: int = 1000,
        burnin: int = 1000,
        n_chains: int = 1,
        dtype: _DTypeName = "float64",
        device: _DeviceRequest = None,
    ) -> None:
        _validate_configuration(n_samples, burnin, n_chains, dtype, device)
        self.n_samples = n_samples
        self.burnin = burnin
        self.n_chains = n_chains
        self.dtype = dtype
        self.device = device

    @property
    def posterior_samples_(self) -> Array:
        """Reconstruct full covariance draws from packed fitted storage."""
        try:
            packed = self.posterior_samples_packed_
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError(
                "posterior_samples_ is available only after fit"
            ) from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    @property
    def phi_samples_(self) -> Array:
        """Reconstruct full local-scale draws from packed fitted storage."""
        try:
            packed = self.phi_samples_packed_
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError("phi_samples_ is available only after fit") from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    def fit(
        self,
        X: ArrayLike,
        *,
        key: Array,
        initial_covariance: ArrayLike | None = None,
    ) -> Self:
        """Fit independent BM chains and publish posterior covariance draws."""
        _validate_key(key)
        _validate_configuration(
            self.n_samples,
            self.burnin,
            self.n_chains,
            self.dtype,
            self.device,
        )
        if self.dtype == "float64" and not _x64_enabled():
            raise RuntimeError(
                "dtype='float64' requires JAX X64 mode. Set JAX_ENABLE_X64=1 "
                "before starting Python."
            )
        target = _resolve_device(self.device)
        dtype = jnp.dtype(self.dtype)

        with jax.default_device(target):
            try:
                raw_x = jnp.asarray(X)
            except (TypeError, ValueError) as error:
                raise TypeError("X must be a real numeric array") from error
            if not jnp.issubdtype(raw_x.dtype, jnp.number) or jnp.issubdtype(
                raw_x.dtype,
                jnp.complexfloating,
            ):
                raise TypeError("X must be a real numeric array")
            x = cast(Array, jax.device_put(jnp.asarray(raw_x, dtype=dtype), target))
            n_observations, dimension = _validate_x(x)
            scatter = x.T @ x

            if initial_covariance is None:
                covariance, n_initial_repairs, initial_variance_floor = (
                    _default_initial_covariance(x)
                )
            else:
                n_initial_repairs = 0
                initial_variance_floor = 0.0
                try:
                    raw_covariance = jnp.asarray(initial_covariance)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "initial_covariance must be a real numeric array"
                    ) from error
                if not jnp.issubdtype(
                    raw_covariance.dtype, jnp.number
                ) or jnp.issubdtype(raw_covariance.dtype, jnp.complexfloating):
                    raise TypeError("initial_covariance must be a real numeric array")
                covariance = cast(
                    Array,
                    jax.device_put(jnp.asarray(raw_covariance, dtype=dtype), target),
                )
            _validate_initial_covariance(covariance, dimension)

            tau1sq = jnp.asarray(
                10_000.0 / (n_observations * dimension**4),
                dtype=dtype,
            )
            initial_state = initialize_bm_state(covariance, tau1sq)
            states = jax.tree.map(
                lambda value: jnp.broadcast_to(
                    value,
                    (self.n_chains, *value.shape),
                ),
                initial_state,
            )
            chain_keys = jax.random.split(
                jax.device_put(key, target),
                self.n_chains,
            )
            result = _compile_bm_chains()(
                chain_keys,
                states,
                scatter,
                _other_indices(dimension, target),
                jnp.asarray(n_observations, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=dtype),
                jnp.asarray(0.5, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
                tau1sq,
                burnin=self.burnin,
                n_samples=self.n_samples,
            )
            result.covariance.block_until_ready()

        rejected = int(jnp.count_nonzero(~result.accepted))
        if rejected:
            first = jnp.argwhere(~result.accepted, size=1)[0]
            chain_index = int(first[0])
            sweep_index = int(first[1])
            suffix = "" if rejected == 1 else "s"
            raise RuntimeError(
                f"BM sampling rejected {rejected} sweep{suffix}; first rejection "
                f"was at chain {chain_index}, sweep {sweep_index}"
            )
        posterior_mean_packed = jnp.mean(result.covariance, axis=(0, 1))
        covariance_mean = unpack_lower_triangle_column_major(
            posterior_mean_packed,
            dimension=dimension,
        )
        diagnostics = BMDiagnostics(
            accepted=result.accepted,
            n_sweeps=self.n_chains * (self.burnin + self.n_samples),
            n_rejected_sweeps=0,
            dtype=str(dtype),
            device=f"{target.platform}:{target.id}",
            n_initial_repairs=n_initial_repairs,
            initial_variance_floor=initial_variance_floor,
        )

        self.posterior_samples_packed_ = result.covariance
        self.phi_samples_packed_ = result.phi
        self.covariance_ = covariance_mean
        self.diagnostics_ = diagnostics
        self.initial_covariance_ = covariance
        self.n_features_in_ = dimension
        self.n_observations_ = n_observations
        self.dtype_ = dtype
        self.device_ = target
        return self


class SBMSPCov(_PosteriorSummariesMixin):
    """Screened beta-mixture sparse covariance estimator.

    X is interpreted like upstream bspcov::sbmspcov: rows are observations,
    columns are variables, and the sampler uses X.T @ X without silently
    centering the data. Center data before calling fit when the mean is unknown.

    Screening runs once per fit, and every Python chain shares the resulting
    fixed support. This differs intentionally from bspcov 1.0.3, whose FNR path
    consumes fresh screening RNG separately for each chain.
    """

    covariance_: Array
    posterior_samples_packed_: Array
    phi_samples_packed_: Array
    diagnostics_: SBMDiagnostics
    initial_covariance_: Array
    screening_mask_: Array
    screening_cutoff_: Array | None
    n_features_in_: int
    n_observations_: int
    dtype_: jnp.dtype
    device_: jax.Device

    def __init__(
        self,
        *,
        n_samples: int = 1000,
        burnin: int = 1000,
        n_chains: int = 1,
        cutoff_method: _CutoffMethod = "fnr",
        fnr_correlation: float | Array = 0.25,
        false_negative_rate: float | Array = 0.05,
        n_cutoff_simulations: int = 1000,
        retained_fraction: float | Array = 0.2,
        dtype: _DTypeName = "float64",
        device: _DeviceRequest = None,
    ) -> None:
        _validate_sbm_configuration(
            n_samples,
            burnin,
            n_chains,
            cutoff_method,
            fnr_correlation,
            false_negative_rate,
            n_cutoff_simulations,
            retained_fraction,
            dtype,
            device,
        )
        self.n_samples = n_samples
        self.burnin = burnin
        self.n_chains = n_chains
        self.cutoff_method = cutoff_method
        self.fnr_correlation = fnr_correlation
        self.false_negative_rate = false_negative_rate
        self.n_cutoff_simulations = n_cutoff_simulations
        self.retained_fraction = retained_fraction
        self.dtype = dtype
        self.device = device

    @property
    def posterior_samples_(self) -> Array:
        """Reconstruct full covariance draws from packed fitted storage."""
        try:
            packed = self.posterior_samples_packed_
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError(
                "posterior_samples_ is available only after fit"
            ) from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    @property
    def phi_samples_(self) -> Array:
        """Reconstruct full local-scale draws from packed fitted storage."""
        try:
            packed = self.phi_samples_packed_
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError("phi_samples_ is available only after fit") from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    def fit(
        self,
        X: ArrayLike,
        *,
        key: Array,
        initial_covariance: ArrayLike | None = None,
    ) -> Self:
        """Screen once, then fit independent SBM chains on the shared support."""
        _validate_key(key)
        _validate_sbm_configuration(
            self.n_samples,
            self.burnin,
            self.n_chains,
            self.cutoff_method,
            self.fnr_correlation,
            self.false_negative_rate,
            self.n_cutoff_simulations,
            self.retained_fraction,
            self.dtype,
            self.device,
        )
        if self.dtype == "float64" and not _x64_enabled():
            raise RuntimeError(
                "dtype='float64' requires JAX X64 mode. Set JAX_ENABLE_X64=1 "
                "before starting Python."
            )
        target = _resolve_device(self.device)
        dtype = jnp.dtype(self.dtype)
        cutoff_method = self.cutoff_method

        with jax.default_device(target):
            try:
                raw_x = jnp.asarray(X)
            except (TypeError, ValueError) as error:
                raise TypeError("X must be a real numeric array") from error
            if not jnp.issubdtype(raw_x.dtype, jnp.number) or jnp.issubdtype(
                raw_x.dtype,
                jnp.complexfloating,
            ):
                raise TypeError("X must be a real numeric array")
            x = cast(Array, jax.device_put(jnp.asarray(raw_x, dtype=dtype), target))
            n_observations, dimension = _validate_x(x)
            scatter = x.T @ x

            if initial_covariance is None:
                covariance, n_initial_repairs, initial_variance_floor = (
                    _default_initial_covariance(x)
                )
            else:
                n_initial_repairs = 0
                initial_variance_floor = 0.0
                try:
                    raw_covariance = jnp.asarray(initial_covariance)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "initial_covariance must be a real numeric array"
                    ) from error
                if not jnp.issubdtype(
                    raw_covariance.dtype, jnp.number
                ) or jnp.issubdtype(raw_covariance.dtype, jnp.complexfloating):
                    raise TypeError("initial_covariance must be a real numeric array")
                covariance = cast(
                    Array,
                    jax.device_put(jnp.asarray(raw_covariance, dtype=dtype), target),
                )
            _validate_initial_covariance(covariance, dimension)

            master_key = cast(Array, jax.device_put(key, target))
            screening_key, sampler_key = jax.random.split(master_key)
            screening_cutoff: Array | None
            if cutoff_method == "fnr":
                scores = pairwise_jeffreys_bayes_factors(x)
                screening_cutoff = estimate_fnr_cutoff(
                    screening_key,
                    n_observations=n_observations,
                    correlation=self.fnr_correlation,
                    false_negative_rate=self.false_negative_rate,
                    n_simulations=self.n_cutoff_simulations,
                    dtype=self.dtype,
                )
                active_mask = fnr_screening_mask(scores, screening_cutoff)
            else:
                screening_cutoff = None
                active_mask = correlation_screening_mask(
                    x,
                    retained_fraction=self.retained_fraction,
                )
            active_mask = cast(
                Array,
                jax.device_put(
                    validate_sbm_active_mask(active_mask, dimension=dimension),
                    target,
                ),
            )

            other_indices = _other_indices(dimension, target)
            structure = prepare_sbm_compact_structure(active_mask, other_indices)
            tau1sq = jnp.log(jnp.asarray(dimension, dtype=dtype)) / jnp.asarray(
                dimension**2 * n_observations,
                dtype=dtype,
            )
            initial_state = initialize_sbm_state(covariance, tau1sq, active_mask)
            supported = active_mask | jnp.eye(dimension, dtype=jnp.bool_)
            screened_without_jitter = jnp.where(supported, covariance, 0.0)
            screening_jitter = float(
                initial_state.covariance[0, 0] - screened_without_jitter[0, 0]
            )
            states = jax.tree.map(
                lambda value: jnp.broadcast_to(
                    value,
                    (self.n_chains, *value.shape),
                ),
                initial_state,
            )
            chain_keys = jax.random.split(sampler_key, self.n_chains)
            result = _compile_sbm_chains()(
                chain_keys,
                states,
                scatter,
                jnp.asarray(n_observations, dtype=jnp.int32),
                jnp.asarray(0.5, dtype=dtype),
                jnp.asarray(0.5, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
                tau1sq,
                structure,
                burnin=self.burnin,
                n_samples=self.n_samples,
            )
            result.covariance.block_until_ready()

        rejected = int(jnp.count_nonzero(~result.accepted))
        if rejected:
            first = jnp.argwhere(~result.accepted, size=1)[0]
            chain_index = int(first[0])
            sweep_index = int(first[1])
            suffix = "" if rejected == 1 else "s"
            raise RuntimeError(
                f"SBM sampling rejected {rejected} sweep{suffix}; first rejection "
                f"was at chain {chain_index}, sweep {sweep_index}"
            )

        posterior_mean_packed = jnp.mean(result.covariance, axis=(0, 1))
        covariance_mean = unpack_lower_triangle_column_major(
            posterior_mean_packed,
            dimension=dimension,
        )
        active_edges = int(jnp.count_nonzero(jnp.tril(active_mask, k=-1)))
        total_edges = dimension * (dimension - 1) // 2
        diagnostics = SBMDiagnostics(
            accepted=result.accepted,
            n_sweeps=self.n_chains * (self.burnin + self.n_samples),
            n_rejected_sweeps=0,
            n_initial_repairs=n_initial_repairs,
            initial_variance_floor=initial_variance_floor,
            screening_jitter=screening_jitter,
            n_active_edges=active_edges,
            n_screened_edges=total_edges - active_edges,
            cutoff_method=cutoff_method,
            dtype=str(dtype),
            device=f"{target.platform}:{target.id}",
        )

        self.posterior_samples_packed_ = result.covariance
        self.phi_samples_packed_ = result.phi
        self.covariance_ = covariance_mean
        self.diagnostics_ = diagnostics
        self.initial_covariance_ = initial_state.covariance
        self.screening_mask_ = active_mask
        self.screening_cutoff_ = screening_cutoff
        self.n_features_in_ = dimension
        self.n_observations_ = n_observations
        self.dtype_ = dtype
        self.device_ = target
        return self
