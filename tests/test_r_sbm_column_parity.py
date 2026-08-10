from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pybspcov.kernels.sbm import sbm_column_parameters

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_csv(name: str, *, dtype: type[np.generic] = np.float64) -> np.ndarray:
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", dtype=dtype, ndmin=1)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(jnp.float64, 1e-12, 1e-12), (jnp.float32, 3e-5, 3e-6)],
)
def test_sbm_column_parameters_match_bspcov_1_0_3(
    dtype: jnp.dtype,
    rtol: float,
    atol: float,
) -> None:
    if dtype == jnp.float64:
        assert jax.config.x64_enabled

    x = jnp.asarray(_load_csv("sbm_column_x.csv"), dtype=dtype)
    inputs = _load_csv("sbm_column_parameters.csv")
    active_mask = jnp.asarray(
        _load_csv("sbm_column_active_mask.csv", dtype=np.int64), dtype=jnp.bool_
    )
    parameters = jax.jit(sbm_column_parameters)(
        covariance=jnp.asarray(_load_csv("sbm_column_covariance.csv"), dtype=dtype),
        precision=jnp.asarray(_load_csv("sbm_column_precision.csv"), dtype=dtype),
        scatter=x.T @ x,
        tau=jnp.asarray(_load_csv("sbm_column_tau.csv"), dtype=dtype),
        active_mask=active_mask,
        column=jnp.asarray(int(inputs[0]), dtype=jnp.int32),
        other_indices=jnp.asarray(
            _load_csv("sbm_column_other_indices.csv", dtype=np.int64),
            dtype=jnp.int32,
        ),
        n_observations=jnp.asarray(inputs[1], dtype=dtype),
        diagonal_rate=jnp.asarray(inputs[2], dtype=dtype),
        gamma=jnp.asarray(inputs[3], dtype=dtype),
    )

    expected_active = _load_csv(
        "sbm_column_expected_active.csv", dtype=np.int64
    ).astype(bool)
    np.testing.assert_array_equal(np.asarray(parameters.active), expected_active)
    assert int(parameters.active_count) == int(expected_active.sum())

    comparisons = (
        (
            parameters.conditional_precision,
            "sbm_column_expected_conditional_precision.csv",
        ),
        (
            parameters.conditional_scatter,
            "sbm_column_expected_conditional_scatter.csv",
        ),
        (parameters.quadratic, "sbm_column_expected_quadratic.csv"),
        (parameters.beta_precision, "sbm_column_expected_beta_precision.csv"),
        (parameters.beta_mean, "sbm_column_expected_beta_mean.csv"),
    )
    for actual, fixture_name in comparisons:
        np.testing.assert_allclose(
            np.asarray(actual),
            _load_csv(fixture_name),
            rtol=rtol,
            atol=atol,
        )

    expected_gamma = _load_csv("sbm_column_expected_gamma_parameters.csv")
    np.testing.assert_allclose(
        np.asarray(
            [parameters.gamma_lambda, parameters.gamma_chi, parameters.gamma_psi]
        ),
        expected_gamma,
        rtol=rtol,
        atol=atol,
    )
