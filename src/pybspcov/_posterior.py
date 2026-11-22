"""Posterior summaries shared by the public estimators."""

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from pybspcov._validation import as_real_array
from pybspcov.kernels.bm import unpack_lower_triangle_column_major


@dataclass(frozen=True)
class PosteriorSummary:
    """Elementwise posterior statistics pooled across fitted chains."""

    mean: Array
    standard_deviation: Array
    probabilities: Array
    quantiles: Array
    n_chains: int
    n_samples_per_chain: int


class PosteriorSummariesMixin:
    posterior_samples_packed_: Array
    n_features_in_: int

    def _require_posterior_samples(self, method_name: str) -> Array:
        try:
            return self.posterior_samples_packed_
        except AttributeError:
            raise AttributeError(f"{method_name} is available only after fit") from None

    def _unpack_fitted(self, packed_name: str, public_name: str) -> Array:
        try:
            packed = getattr(self, packed_name)
            dimension = self.n_features_in_
        except AttributeError:
            raise AttributeError(f"{public_name} is available only after fit") from None
        return unpack_lower_triangle_column_major(packed, dimension=dimension)

    @staticmethod
    def _probabilities(probs: ArrayLike | Sequence[float], packed: Array) -> Array:
        values = as_real_array(probs, "probabilities must contain real numbers")
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

    @property
    def posterior_samples_(self) -> Array:
        """Reconstruct full covariance draws from packed fitted storage."""
        return self._unpack_fitted("posterior_samples_packed_", "posterior_samples_")

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
