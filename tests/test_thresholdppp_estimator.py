import jax
import jax.numpy as jnp
import pytest

from pybspcov import ThresholdPPP


def _centered_case() -> jax.Array:
    return jnp.asarray(
        [
            [-1.5, -0.5, 0.5],
            [-0.5, 1.5, -1.5],
            [0.5, -1.5, 1.5],
            [1.5, 0.5, -0.5],
        ],
        dtype=jnp.float32,
    )


def test_thresholdppp_fit_publishes_sparse_positive_definite_draws() -> None:
    model = ThresholdPPP(
        threshold=10.0,
        method="hard",
        epsilon=0.05,
        n_samples=4,
        n_chains=2,
        dtype="float32",
    ).fit(_centered_case(), key=jax.random.key(211))

    assert model.posterior_samples_packed_.shape == (2, 4, 6)
    assert model.posterior_samples_.shape == (2, 4, 3, 3)
    off_diagonal = ~jnp.eye(3, dtype=jnp.bool_)
    assert jnp.all(model.posterior_samples_[..., off_diagonal] == 0.0)
    assert jnp.all(jnp.linalg.eigvalsh(model.posterior_samples_) >= 0.05 - 2e-6)
    assert jnp.allclose(model.covariance_, model.estimate())
    assert model.to_arviz().posterior["covariance"].shape == (2, 4, 3, 3)


def test_thresholdppp_uses_upstream_prior_defaults() -> None:
    x = _centered_case()
    model = ThresholdPPP(
        threshold=0.1,
        epsilon=0.0,
        n_samples=1,
        dtype="float32",
    ).fit(x, key=jax.random.key(223))

    assert float(model.prior_df_) == 4.0
    assert jnp.array_equal(model.prior_scale_, jnp.eye(3, dtype=jnp.float32))
    assert jnp.allclose(model.posterior_scale_, x.T @ x + jnp.eye(3))


@pytest.mark.parametrize("method", ["adaptive", "", 1])
def test_thresholdppp_rejects_unknown_method(method: object) -> None:
    with pytest.raises((TypeError, ValueError), match="method"):
        ThresholdPPP(
            threshold=0.1,
            method=method,  # type: ignore[arg-type]
            n_samples=1,
            dtype="float32",
        )


@pytest.mark.parametrize("threshold", [-0.1, float("inf"), "small", [0.1]])
def test_thresholdppp_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError), match="threshold"):
        ThresholdPPP(
            threshold=threshold,  # type: ignore[arg-type]
            n_samples=1,
            dtype="float32",
        )


def test_thresholdppp_is_reproducible_for_one_key() -> None:
    configuration = {
        "threshold": 0.2,
        "method": "soft",
        "epsilon": 0.01,
        "n_samples": 3,
        "n_chains": 2,
        "dtype": "float32",
    }
    first = ThresholdPPP(**configuration).fit(
        _centered_case(), key=jax.random.key(227)
    )
    second = ThresholdPPP(**configuration).fit(
        _centered_case(), key=jax.random.key(227)
    )

    assert jnp.array_equal(
        first.posterior_samples_packed_, second.posterior_samples_packed_
    )
