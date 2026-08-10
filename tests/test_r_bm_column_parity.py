from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pybspcov.kernels.bm import bm_column_parameters

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_csv(name: str, *, dtype: type[np.generic] = np.float64) -> np.ndarray:
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", dtype=dtype, ndmin=1)


def test_bm_column_parameters_match_bspcov_1_0_3() -> None:
    assert jax.config.x64_enabled
    x = jnp.asarray(_load_csv("bm_x.csv"), dtype=jnp.float64)
    inputs = _load_csv("bm_parameters.csv")
    parameters = jax.jit(bm_column_parameters)(
        covariance=jnp.asarray(_load_csv("bm_covariance.csv"), dtype=jnp.float64),
        precision=jnp.asarray(_load_csv("bm_precision.csv"), dtype=jnp.float64),
        scatter=x.T @ x,
        tau=jnp.asarray(_load_csv("bm_tau.csv"), dtype=jnp.float64),
        column=jnp.asarray(int(inputs[0]), dtype=jnp.int32),
        other_indices=jnp.asarray(
            _load_csv("bm_other_indices.csv", dtype=np.int64), dtype=jnp.int32
        ),
        n_observations=jnp.asarray(inputs[1], dtype=jnp.float64),
        diagonal_rate=jnp.asarray(inputs[2], dtype=jnp.float64),
        gamma=jnp.asarray(inputs[3], dtype=jnp.float64),
    )

    expected_gamma = _load_csv("bm_expected_gamma_parameters.csv")
    comparisons = (
        (
            parameters.conditional_precision,
            "bm_expected_conditional_precision.csv",
        ),
        (parameters.conditional_scatter, "bm_expected_conditional_scatter.csv"),
        (parameters.quadratic, "bm_expected_quadratic.csv"),
        (parameters.beta_precision, "bm_expected_beta_precision.csv"),
        (parameters.beta_mean, "bm_expected_beta_mean.csv"),
    )
    for actual, fixture_name in comparisons:
        np.testing.assert_allclose(
            np.asarray(actual),
            _load_csv(fixture_name),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        np.asarray(
            [parameters.gamma_lambda, parameters.gamma_chi, parameters.gamma_psi]
        ),
        expected_gamma,
        rtol=1e-12,
        atol=1e-12,
    )
