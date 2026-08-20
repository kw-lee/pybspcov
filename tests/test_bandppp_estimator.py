import jax
import jax.numpy as jnp
import pytest

from pybspcov import BandPPP


def _centered_case(dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(
        [
            [-1.5, -0.5, 0.5],
            [-0.5, 1.5, -1.5],
            [0.5, -1.5, 1.5],
            [1.5, 0.5, -0.5],
        ],
        dtype=dtype,
    )


def test_bandppp_fit_publishes_banded_positive_definite_draws() -> None:
    model = BandPPP(
        bandwidth=1,
        epsilon=0.05,
        n_samples=4,
        n_chains=2,
        dtype="float32",
    )

    fitted = model.fit(_centered_case(jnp.float32), key=jax.random.key(101))

    assert fitted is model
    assert model.posterior_samples_packed_.shape == (2, 4, 6)
    assert model.posterior_samples_.shape == (2, 4, 3, 3)
    assert jnp.array_equal(model.posterior_samples_[..., 0, 2], jnp.zeros((2, 4)))
    assert jnp.all(jnp.linalg.eigvalsh(model.posterior_samples_) >= 0.05 - 2e-6)
    assert jnp.allclose(
        model.covariance_, jnp.mean(model.posterior_samples_, axis=(0, 1))
    )
    assert model.summary().n_chains == 2
    assert model.summary().n_samples_per_chain == 4
    arviz_covariance = model.to_arviz().posterior["covariance"]
    assert arviz_covariance.dims == ("chain", "draw", "row", "column")
    assert arviz_covariance.shape == (2, 4, 3, 3)


@pytest.mark.parametrize("epsilon", [-0.1, float("inf"), "small", [0.1]])
def test_bandppp_rejects_invalid_epsilon(epsilon: object) -> None:
    with pytest.raises((TypeError, ValueError), match="epsilon"):
        BandPPP(
            bandwidth=1,
            epsilon=epsilon,  # type: ignore[arg-type]
            n_samples=1,
            dtype="float32",
        )


def test_bandppp_revalidates_mutated_epsilon_before_fit() -> None:
    model = BandPPP(
        bandwidth=1,
        epsilon=0.1,
        n_samples=1,
        dtype="float32",
    )
    model.epsilon = -1.0

    with pytest.raises(ValueError, match="epsilon"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(103))


def test_bandppp_uses_upstream_r_prior_and_epsilon_defaults() -> None:
    x = _centered_case(jnp.float32)
    model = BandPPP(
        bandwidth=2,
        n_samples=1,
        dtype="float32",
    )

    model.fit(x, key=jax.random.key(107))

    assert jnp.array_equal(model.prior_scale_, jnp.eye(3, dtype=jnp.float32))
    assert float(model.prior_df_) == 5.0
    assert jnp.allclose(model.posterior_scale_, x.T @ x + jnp.eye(3))
    assert float(model.posterior_df_) == 9.0
    assert float(model.epsilon_) == pytest.approx(
        float((jnp.log(2.0) ** 2) * (2.0 + jnp.log(3.0)) / 4.0),
    )


@pytest.mark.parametrize("prior_df", [2.0, float("inf"), "wide", [5.0]])
def test_bandppp_rejects_invalid_prior_degrees_of_freedom(
    prior_df: object,
) -> None:
    model = BandPPP(
        bandwidth=1,
        prior_df=prior_df,  # type: ignore[arg-type]
        n_samples=1,
        dtype="float32",
    )

    with pytest.raises((TypeError, ValueError), match="prior_df"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(109))


@pytest.mark.parametrize(
    "prior_scale",
    [
        jnp.eye(2),
        jnp.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        jnp.diag(jnp.asarray([1.0, 0.0, 1.0])),
        jnp.diag(jnp.asarray([1.0, jnp.inf, 1.0])),
    ],
)
def test_bandppp_rejects_invalid_prior_scale(prior_scale: jax.Array) -> None:
    model = BandPPP(
        bandwidth=1,
        prior_scale=prior_scale,
        n_samples=1,
        dtype="float32",
    )

    with pytest.raises(ValueError, match="prior_scale"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(113))


@pytest.mark.parametrize(
    "prior_scale",
    [jnp.eye(3, dtype=jnp.complex64), jnp.eye(3, dtype=jnp.bool_)],
)
def test_bandppp_prior_scale_must_be_real_numeric(prior_scale: jax.Array) -> None:
    model = BandPPP(
        bandwidth=1,
        prior_scale=prior_scale,
        n_samples=1,
        dtype="float32",
    )

    with pytest.raises(TypeError, match="prior_scale must be a real numeric array"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(127))
