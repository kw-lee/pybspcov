"""Public estimators that orchestrate the pure JAX sampling kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal, NamedTuple, Self, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from pybspcov._posterior import PosteriorSummariesMixin, PosteriorSummary
from pybspcov._validation import (
    as_real_array,
    resolve_prior_df,
    validate_bounded_scalar,
    validate_configuration,
    validate_integer,
    validate_key,
    validate_nonnegative_scalar,
    validate_optional_epsilon,
    validate_positive_definite,
    validate_threshold_method,
    validate_x,
)
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
type _ScreeningScope = Literal["fit", "chain"]
type _BMRunner = Callable[..., BMPackedChainResult]
type _SBMRunner = Callable[..., SBMPackedChainResult]
type _BandPPPRunner = Callable[..., tuple[Array, Array]]
type _ThresholdPPPRunner = Callable[..., tuple[Array, Array]]

__all__ = [
    "BMDiagnostics",
    "BMSPCov",
    "BandPPP",
    "PosteriorSummary",
    "SBMDiagnostics",
    "SBMSPCov",
    "ThresholdPPP",
]


class _InverseWishartPosterior(NamedTuple):
    prior_scale: Array
    prior_df: Array
    posterior_scale: Array
    posterior_df: Array
    n_observations: int
    dimension: int


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
    screening_jitter: float | tuple[float, ...]
    n_active_edges: int | tuple[int, ...]
    n_screened_edges: int | tuple[int, ...]
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


def _other_indices(dimension: int, device: jax.Device) -> Array:
    rows = [
        [index for index in range(dimension) if index != column]
        for column in range(dimension)
    ]
    return cast(Array, jax.device_put(jnp.asarray(rows, dtype=jnp.int32), device))


def _default_initial_covariance(x: Array) -> tuple[Array, int, float]:
    variances = jnp.var(x, axis=0, ddof=1)
    scale = jnp.maximum(jnp.max(jnp.abs(variances)), jnp.asarray(1.0, x.dtype))
    epsilon = 1.1920928955078125e-7 if x.dtype == jnp.float32 else 2.220446049250313e-16
    floor = jnp.asarray(epsilon, x.dtype) * scale
    repairs = int(jnp.count_nonzero(variances < floor))
    return jnp.diag(jnp.maximum(variances, floor)), repairs, float(floor)


def _resolve_fit_target(
    dtype_name: _DTypeName,
    device: _DeviceRequest,
) -> tuple[jax.Device, jnp.dtype]:
    if dtype_name == "float64" and not _x64_enabled():
        raise RuntimeError(
            "dtype='float64' requires JAX X64 mode. Set JAX_ENABLE_X64=1 "
            "before starting Python."
        )
    return _resolve_device(device), jnp.dtype(dtype_name)


def _coerce_x(
    X: ArrayLike,
    dtype: jnp.dtype,
    target: jax.Device,
) -> tuple[Array, int, int]:
    raw_x = as_real_array(X, "X must be a real numeric array")
    x = cast(Array, jax.device_put(jnp.asarray(raw_x, dtype=dtype), target))
    n_observations, dimension = validate_x(x)
    return x, n_observations, dimension


def _resolve_initial_covariance(
    initial_covariance: ArrayLike | None,
    x: Array,
    dimension: int,
    target: jax.Device,
) -> tuple[Array, int, float]:
    if initial_covariance is None:
        covariance, n_initial_repairs, initial_variance_floor = (
            _default_initial_covariance(x)
        )
    else:
        n_initial_repairs = 0
        initial_variance_floor = 0.0
        raw_covariance = as_real_array(
            initial_covariance,
            "initial_covariance must be a real numeric array",
        )
        covariance = cast(
            Array,
            jax.device_put(jnp.asarray(raw_covariance, dtype=x.dtype), target),
        )
    validate_positive_definite("initial_covariance", covariance, dimension)
    return covariance, n_initial_repairs, initial_variance_floor


def _inverse_wishart_posterior(
    X: ArrayLike,
    prior_scale: ArrayLike | None,
    prior_df: float | Array | None,
    *,
    default_prior_df_offset: int,
    dtype: jnp.dtype,
    target: jax.Device,
) -> _InverseWishartPosterior:
    x, n_observations, dimension = _coerce_x(X, dtype, target)
    if prior_scale is None:
        scale = jnp.eye(dimension, dtype=dtype)
    else:
        scale = jnp.asarray(
            as_real_array(prior_scale, "prior_scale must be a real numeric array"),
            dtype=dtype,
        )
    validate_positive_definite("prior_scale", scale, dimension)
    df = resolve_prior_df(
        prior_df,
        default=dimension + default_prior_df_offset,
        dimension=dimension,
        dtype=dtype,
    )
    return _InverseWishartPosterior(
        prior_scale=scale,
        prior_df=df,
        posterior_scale=x.T @ x + scale,
        posterior_df=df + n_observations,
        n_observations=n_observations,
        dimension=dimension,
    )


def _raise_on_rejected_sweeps(sampler: str, accepted: Array) -> None:
    rejected = int(jnp.count_nonzero(~accepted))
    if rejected:
        first = jnp.argwhere(~accepted, size=1)[0]
        chain_index = int(first[0])
        sweep_index = int(first[1])
        suffix = "" if rejected == 1 else "s"
        raise RuntimeError(
            f"{sampler} sampling rejected {rejected} sweep{suffix}; first rejection "
            f"was at chain {chain_index}, sweep {sweep_index}"
        )


class _PostProcessedPosterior(PosteriorSummariesMixin):
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

    def _publish(
        self,
        packed: Array,
        adjusted: Array,
        epsilon: Array,
        posterior: _InverseWishartPosterior,
        dtype: jnp.dtype,
        target: jax.Device,
    ) -> None:
        self.posterior_samples_packed_ = packed
        self.covariance_ = unpack_lower_triangle_column_major(
            jnp.mean(packed, axis=(0, 1)),
            dimension=posterior.dimension,
        )
        self.adjusted_draws_ = adjusted
        self.epsilon_ = epsilon
        self.prior_scale_ = posterior.prior_scale
        self.prior_df_ = posterior.prior_df
        self.posterior_scale_ = posterior.posterior_scale
        self.posterior_df_ = posterior.posterior_df
        self.n_features_in_ = posterior.dimension
        self.n_observations_ = posterior.n_observations
        self.dtype_ = dtype
        self.device_ = target


class _MixtureShrinkagePosterior(PosteriorSummariesMixin):
    phi_samples_packed_: Array

    @property
    def phi_samples_(self) -> Array:
        """Reconstruct full local-scale draws from packed fitted storage."""
        return self._unpack_fitted("phi_samples_packed_", "phi_samples_")


class BandPPP(_PostProcessedPosterior):
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
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        self.prior_scale = prior_scale
        self.prior_df = prior_df
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.dtype = dtype
        self.device = device
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        validate_integer("bandwidth", self.bandwidth, minimum=1)
        validate_configuration(
            self.n_samples, 0, self.n_chains, self.dtype, self.device
        )
        validate_optional_epsilon(self.epsilon)

    def fit(self, X: ArrayLike, *, key: Array) -> Self:
        """Draw the inverse-Wishart posterior and apply BandPPP processing."""
        validate_key(key)
        self._validate_parameters()
        target, dtype = _resolve_fit_target(self.dtype, self.device)

        with jax.default_device(target):
            posterior = _inverse_wishart_posterior(
                X,
                self.prior_scale,
                self.prior_df,
                default_prior_df_offset=self.bandwidth,
                dtype=dtype,
                target=target,
            )
            epsilon = jnp.asarray(
                (jnp.log(self.bandwidth) ** 2)
                * (self.bandwidth + jnp.log(posterior.dimension))
                / posterior.n_observations
                if self.epsilon is None
                else self.epsilon,
                dtype=dtype,
            )
            chain_keys = jax.random.split(jax.device_put(key, target), self.n_chains)
            packed, adjusted = _compile_bandppp_chains()(
                chain_keys,
                posterior.posterior_scale,
                posterior.posterior_df,
                jnp.asarray(self.bandwidth, dtype=jnp.int32),
                epsilon,
                n_samples=self.n_samples,
            )
            packed.block_until_ready()

        self._publish(packed, adjusted, epsilon, posterior, dtype, target)
        return self


class ThresholdPPP(_PostProcessedPosterior):
    """Post-processed posterior estimator for a sparse covariance matrix."""

    threshold_: Array

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
        self.threshold = threshold
        self.method = method
        self.epsilon = epsilon
        self.prior_scale = prior_scale
        self.prior_df = prior_df
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.dtype = dtype
        self.device = device
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        validate_nonnegative_scalar("threshold", self.threshold)
        validate_threshold_method(self.method)
        validate_optional_epsilon(self.epsilon)
        validate_configuration(
            self.n_samples, 0, self.n_chains, self.dtype, self.device
        )

    def fit(self, X: ArrayLike, *, key: Array) -> Self:
        """Draw inverse-Wishart samples and apply sparse post-processing."""
        validate_key(key)
        self._validate_parameters()
        target, dtype = _resolve_fit_target(self.dtype, self.device)

        with jax.default_device(target):
            posterior = _inverse_wishart_posterior(
                X,
                self.prior_scale,
                self.prior_df,
                default_prior_df_offset=1,
                dtype=dtype,
                target=target,
            )
            threshold = jnp.asarray(self.threshold, dtype=dtype)
            epsilon = jnp.asarray(self.epsilon, dtype=dtype)
            chain_keys = jax.random.split(jax.device_put(key, target), self.n_chains)
            packed, adjusted = _compile_thresholdppp_chains()(
                chain_keys,
                posterior.posterior_scale,
                posterior.posterior_df,
                threshold,
                epsilon,
                method=self.method,
                n_samples=self.n_samples,
            )
            packed.block_until_ready()

        self._publish(packed, adjusted, epsilon, posterior, dtype, target)
        self.threshold_ = threshold
        return self


class BMSPCov(_MixtureShrinkagePosterior):
    """Beta-mixture sparse covariance estimator.

    ``X`` is interpreted like upstream ``bspcov::bmspcov``: rows are
    observations, columns are variables, and the sampler uses ``X.T @ X``
    without silently centering the data. Center data before calling
    :meth:`fit` when the mean is unknown.

    Examples:
        Fit a short float32 chain on the CPU and inspect the stable output
        shape. Longer runs are required for inference.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> X = jnp.concatenate((jnp.eye(3), -jnp.eye(3))).astype(jnp.float32)
        >>> model = BMSPCov(
        ...     n_samples=2, burnin=1, dtype="float32", device="cpu"
        ... ).fit(X, key=jax.random.key(17))
        >>> model.estimate().shape
        (3, 3)

        See the :doc:`cached benchmark summary <benchmarks>` for measured
        CPU and GPU results; Sphinx does not rerun those measurements.
    """

    covariance_: Array
    posterior_samples_packed_: Array
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
        self.n_samples = n_samples
        self.burnin = burnin
        self.n_chains = n_chains
        self.dtype = dtype
        self.device = device
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        validate_configuration(
            self.n_samples, self.burnin, self.n_chains, self.dtype, self.device
        )

    def fit(
        self,
        X: ArrayLike,
        *,
        key: Array,
        initial_covariance: ArrayLike | None = None,
    ) -> Self:
        """Fit independent BM chains and publish posterior covariance draws."""
        validate_key(key)
        self._validate_parameters()
        target, dtype = _resolve_fit_target(self.dtype, self.device)

        with jax.default_device(target):
            x, n_observations, dimension = _coerce_x(X, dtype, target)
            scatter = x.T @ x
            covariance, n_initial_repairs, initial_variance_floor = (
                _resolve_initial_covariance(initial_covariance, x, dimension, target)
            )

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

        _raise_on_rejected_sweeps("BM", result.accepted)
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


class SBMSPCov(_MixtureShrinkagePosterior):
    """Screened beta-mixture sparse covariance estimator.

    X is interpreted like upstream bspcov::sbmspcov: rows are observations,
    columns are variables, and the sampler uses X.T @ X without silently
    centering the data. Center data before calling fit when the mean is unknown.

    By default, screening runs once per fit and all chains share the support.
    Set ``screening_scope="chain"`` to draw and apply an independent FNR
    screening cutoff for each chain, matching bspcov 1.0.3 semantics.

    Examples:
        Correlation screening avoids the stochastic FNR cutoff calculation in
        this short output-shape example.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> X = jnp.concatenate((jnp.eye(3), -jnp.eye(3))).astype(jnp.float32)
        >>> model = SBMSPCov(
        ...     n_samples=2,
        ...     burnin=1,
        ...     cutoff_method="correlation",
        ...     retained_fraction=0.5,
        ...     dtype="float32",
        ...     device="cpu",
        ... ).fit(X, key=jax.random.key(101))
        >>> model.estimate().shape
        (3, 3)

        See the :doc:`cached benchmark summary <benchmarks>` for measured
        CPU and GPU results; Sphinx does not rerun those measurements.
    """

    covariance_: Array
    posterior_samples_packed_: Array
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
        screening_scope: _ScreeningScope = "fit",
        fnr_correlation: float | Array = 0.25,
        false_negative_rate: float | Array = 0.05,
        n_cutoff_simulations: int = 1000,
        retained_fraction: float | Array = 0.2,
        dtype: _DTypeName = "float64",
        device: _DeviceRequest = None,
    ) -> None:
        self.n_samples = n_samples
        self.burnin = burnin
        self.n_chains = n_chains
        self.cutoff_method = cutoff_method
        self.screening_scope = screening_scope
        self.fnr_correlation = fnr_correlation
        self.false_negative_rate = false_negative_rate
        self.n_cutoff_simulations = n_cutoff_simulations
        self.retained_fraction = retained_fraction
        self.dtype = dtype
        self.device = device
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        validate_configuration(
            self.n_samples, self.burnin, self.n_chains, self.dtype, self.device
        )
        validate_integer("n_cutoff_simulations", self.n_cutoff_simulations, minimum=1)
        if self.cutoff_method not in {"fnr", "correlation"}:
            raise ValueError("cutoff_method must be 'fnr' or 'correlation'")
        if self.screening_scope not in {"fit", "chain"}:
            raise ValueError("screening_scope must be 'fit' or 'chain'")
        validate_bounded_scalar(
            "fnr_correlation", self.fnr_correlation, lower=-1.0, upper=1.0
        )
        validate_bounded_scalar(
            "false_negative_rate", self.false_negative_rate, lower=0.0, upper=1.0
        )
        validate_bounded_scalar(
            "retained_fraction", self.retained_fraction, lower=0.0, upper=1.0
        )

    def fit(
        self,
        X: ArrayLike,
        *,
        key: Array,
        initial_covariance: ArrayLike | None = None,
    ) -> Self:
        """Screen and fit independent SBM chains on the requested support scope."""
        validate_key(key)
        self._validate_parameters()
        target, dtype = _resolve_fit_target(self.dtype, self.device)
        cutoff_method = self.cutoff_method

        with jax.default_device(target):
            x, n_observations, dimension = _coerce_x(X, dtype, target)
            scatter = x.T @ x
            covariance, n_initial_repairs, initial_variance_floor = (
                _resolve_initial_covariance(initial_covariance, x, dimension, target)
            )

            master_key = cast(Array, jax.device_put(key, target))
            screening_key, sampler_key = jax.random.split(master_key)
            if self.screening_scope == "fit":
                screening_keys = (screening_key,)
            else:
                screening_keys = tuple(jax.random.split(screening_key, self.n_chains))

            screening_cutoff: Array | None
            if cutoff_method == "fnr":
                scores = pairwise_jeffreys_bayes_factors(x)
                cutoffs = tuple(
                    estimate_fnr_cutoff(
                        chain_screening_key,
                        n_observations=n_observations,
                        correlation=self.fnr_correlation,
                        false_negative_rate=self.false_negative_rate,
                        n_simulations=self.n_cutoff_simulations,
                        dtype=self.dtype,
                    )
                    for chain_screening_key in screening_keys
                )
                active_masks = tuple(
                    fnr_screening_mask(scores, cutoff) for cutoff in cutoffs
                )
                screening_cutoff = (
                    cutoffs[0] if self.screening_scope == "fit" else jnp.stack(cutoffs)
                )
            else:
                screening_cutoff = None
                shared_mask = correlation_screening_mask(
                    x,
                    retained_fraction=self.retained_fraction,
                )
                active_masks = (
                    (shared_mask,)
                    if self.screening_scope == "fit"
                    else (shared_mask,) * self.n_chains
                )

            active_masks = tuple(
                cast(
                    Array,
                    jax.device_put(
                        validate_sbm_active_mask(mask, dimension=dimension),
                        target,
                    ),
                )
                for mask in active_masks
            )
            screening_mask = (
                active_masks[0]
                if self.screening_scope == "fit"
                else jnp.stack(active_masks)
            )

            other_indices = _other_indices(dimension, target)
            tau1sq = jnp.log(jnp.asarray(dimension, dtype=dtype)) / jnp.asarray(
                dimension**2 * n_observations,
                dtype=dtype,
            )
            initial_states = tuple(
                initialize_sbm_state(covariance, tau1sq, mask) for mask in active_masks
            )
            structures = tuple(
                prepare_sbm_compact_structure(mask, other_indices)
                for mask in active_masks
            )
            screening_jitters = tuple(
                float(
                    state.covariance[0, 0]
                    - jnp.where(
                        mask | jnp.eye(dimension, dtype=jnp.bool_),
                        covariance,
                        0.0,
                    )[0, 0]
                )
                for state, mask in zip(initial_states, active_masks, strict=True)
            )
            chain_keys = jax.random.split(sampler_key, self.n_chains)
            if self.screening_scope == "fit":
                states = jax.tree.map(
                    lambda value: jnp.broadcast_to(
                        value,
                        (self.n_chains, *value.shape),
                    ),
                    initial_states[0],
                )
                result = _compile_sbm_chains()(
                    chain_keys,
                    states,
                    scatter,
                    jnp.asarray(n_observations, dtype=jnp.int32),
                    jnp.asarray(0.5, dtype=dtype),
                    jnp.asarray(0.5, dtype=dtype),
                    jnp.asarray(1.0, dtype=dtype),
                    tau1sq,
                    structures[0],
                    burnin=self.burnin,
                    n_samples=self.n_samples,
                )
                fitted_initial_covariance = initial_states[0].covariance
            else:
                chain_results = []
                for chain_index, (state, structure) in enumerate(
                    zip(initial_states, structures, strict=True)
                ):
                    batched_state = jax.tree.map(
                        lambda value: value[jnp.newaxis, ...],
                        state,
                    )
                    chain_results.append(
                        _compile_sbm_chains()(
                            chain_keys[chain_index : chain_index + 1],
                            batched_state,
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
                    )
                result = jax.tree.map(
                    lambda *values: jnp.concatenate(values, axis=0),
                    *chain_results,
                )
                fitted_initial_covariance = jnp.stack(
                    [state.covariance for state in initial_states]
                )
            result.covariance.block_until_ready()

        _raise_on_rejected_sweeps("SBM", result.accepted)

        posterior_mean_packed = jnp.mean(result.covariance, axis=(0, 1))
        covariance_mean = unpack_lower_triangle_column_major(
            posterior_mean_packed,
            dimension=dimension,
        )
        active_edge_counts = tuple(
            int(jnp.count_nonzero(jnp.tril(mask, k=-1))) for mask in active_masks
        )
        total_edges = dimension * (dimension - 1) // 2
        screened_edge_counts = tuple(
            total_edges - count for count in active_edge_counts
        )
        if self.screening_scope == "fit":
            diagnostic_jitter: float | tuple[float, ...] = screening_jitters[0]
            diagnostic_active_edges: int | tuple[int, ...] = active_edge_counts[0]
            diagnostic_screened_edges: int | tuple[int, ...] = screened_edge_counts[0]
        else:
            diagnostic_jitter = screening_jitters
            diagnostic_active_edges = active_edge_counts
            diagnostic_screened_edges = screened_edge_counts
        diagnostics = SBMDiagnostics(
            accepted=result.accepted,
            n_sweeps=self.n_chains * (self.burnin + self.n_samples),
            n_rejected_sweeps=0,
            n_initial_repairs=n_initial_repairs,
            initial_variance_floor=initial_variance_floor,
            screening_jitter=diagnostic_jitter,
            n_active_edges=diagnostic_active_edges,
            n_screened_edges=diagnostic_screened_edges,
            cutoff_method=cutoff_method,
            dtype=str(dtype),
            device=f"{target.platform}:{target.id}",
        )

        self.posterior_samples_packed_ = result.covariance
        self.phi_samples_packed_ = result.phi
        self.covariance_ = covariance_mean
        self.diagnostics_ = diagnostics
        self.initial_covariance_ = fitted_initial_covariance
        self.screening_mask_ = screening_mask
        self.screening_cutoff_ = screening_cutoff
        self.n_features_in_ = dimension
        self.n_observations_ = n_observations
        self.dtype_ = dtype
        self.device_ = target
        return self
