"""Argument validation shared by the public estimators."""

import jax
import jax.numpy as jnp
from jax import Array


def as_real_array(value: object, message: str) -> Array:
    """Convert a value to a JAX array and reject non-real element types."""
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(message) from error
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype,
        jnp.complexfloating,
    ):
        raise TypeError(message)
    return array


def as_real_scalar(name: str, value: object) -> Array:
    """Convert a value to a zero-dimensional real JAX array."""
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
    return scalar


def validate_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be positive")


def validate_configuration(
    n_samples: int,
    burnin: int,
    n_chains: int,
    dtype: object,
    device: object,
) -> None:
    validate_integer("n_samples", n_samples, minimum=1)
    validate_integer("burnin", burnin, minimum=0)
    validate_integer("n_chains", n_chains, minimum=1)
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if isinstance(device, str) and device not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be 'cpu', 'gpu', or 'cuda'")
    if device is not None and not isinstance(device, (str, jax.Device)):
        raise TypeError("device must be 'cpu', 'gpu', 'cuda', a JAX device, or None")


def validate_bounded_scalar(
    name: str,
    value: object,
    *,
    lower: float,
    upper: float,
) -> None:
    scalar = as_real_scalar(name, value)
    if not bool(jnp.isfinite(scalar)) or not bool(
        (scalar >= lower) & (scalar <= upper)
    ):
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")


def validate_nonnegative_scalar(name: str, value: object) -> None:
    scalar = as_real_scalar(name, value)
    if not bool(jnp.isfinite(scalar)) or not bool(scalar >= 0.0):
        raise ValueError(f"{name} must be finite and non-negative")


def validate_optional_epsilon(value: object) -> None:
    if value is not None:
        validate_nonnegative_scalar("epsilon", value)


def validate_threshold_method(method: object) -> None:
    if not isinstance(method, str):
        raise TypeError("method must be 'hard' or 'soft'")
    if method not in {"hard", "soft"}:
        raise ValueError("method must be 'hard' or 'soft'")


def validate_key(key: Array) -> None:
    if not isinstance(key, jax.Array) or not jnp.issubdtype(
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise TypeError("key must be a typed JAX key from jax.random.key")
    if key.ndim != 0:
        raise ValueError("key must be a scalar key; pass one unsplit master key")


def validate_x(x: Array) -> tuple[int, int]:
    """Validate a data matrix and return ``(n_observations, dimension)``."""
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


def validate_positive_definite(name: str, matrix: Array, dimension: int) -> None:
    """Require a finite, symmetric, positive-definite ``p`` by ``p`` matrix."""
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape "
            f"({dimension}, {dimension}); received {matrix.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values")
    tolerance = 1e-5 if matrix.dtype == jnp.float32 else 1e-12
    if not bool(jnp.allclose(matrix, matrix.T, rtol=tolerance, atol=tolerance)):
        raise ValueError(f"{name} must be symmetric")
    if not bool(jnp.all(jnp.linalg.eigvalsh(matrix) > 0.0)):
        raise ValueError(f"{name} must be positive definite")


def resolve_prior_df(
    value: object,
    *,
    default: int,
    dimension: int,
    dtype: jnp.dtype,
) -> Array:
    """Return validated inverse-Wishart degrees of freedom in ``dtype``."""
    raw = as_real_scalar("prior_df", default if value is None else value)
    scalar = jnp.asarray(raw, dtype=dtype)
    if not bool(jnp.isfinite(scalar)):
        raise ValueError("prior_df must be finite")
    if not bool(scalar > dimension - 1):
        raise ValueError(f"prior_df must be greater than p - 1 ({dimension - 1})")
    return scalar
