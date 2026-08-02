"""Public estimators that orchestrate the pure JAX sampling kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal, Self, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from pybspcov.kernels.bm import (
    BMPackedChainResult,
    initialize_bm_state,
    sample_bm_packed_chains,
    unpack_lower_triangle_column_major,
)

type _DTypeName = Literal["float32", "float64"]
type _DeviceName = Literal["cpu", "gpu", "cuda"]
type _DeviceRequest = _DeviceName | jax.Device | None
type _BMRunner = Callable[..., BMPackedChainResult]


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


class BMSPCov:
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
