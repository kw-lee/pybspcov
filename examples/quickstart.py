"""Fit the two pybspcov estimators on a small centered data matrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pybspcov import BMSPCov, SBMSPCov


def main() -> None:
    observations = jnp.asarray(
        [
            [-1.0, 0.5, 0.2],
            [-0.4, -0.7, 0.1],
            [0.2, 0.1, -0.8],
            [0.5, -0.2, 0.6],
            [0.9, 0.4, -0.3],
            [-0.2, -0.1, 0.2],
        ],
        dtype=jnp.float32,
    )

    bm = BMSPCov(
        n_samples=2,
        burnin=1,
        dtype="float32",
        device="cpu",
    ).fit(observations, key=jax.random.key(17))
    print(f"BMSPCov covariance shape: {bm.covariance_.shape}")
    print(f"BMSPCov device: {bm.device_.platform}:{bm.device_.id}")

    sbm = SBMSPCov(
        n_samples=2,
        burnin=1,
        cutoff_method="correlation",
        retained_fraction=0.5,
        dtype="float32",
        device="cpu",
    ).fit(observations, key=jax.random.key(101))
    print(f"SBMSPCov covariance shape: {sbm.covariance_.shape}")
    print(f"SBMSPCov active edges: {sbm.diagnostics_.n_active_edges}")
    print(f"SBMSPCov device: {sbm.device_.platform}:{sbm.device_.id}")


if __name__ == "__main__":
    main()
