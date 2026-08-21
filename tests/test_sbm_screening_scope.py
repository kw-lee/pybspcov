import jax
import jax.numpy as jnp
import pytest

import pybspcov.estimators as estimator_module
from pybspcov import SBMSPCov


def _case() -> jax.Array:
    return jnp.asarray(
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


def test_sbmspcov_validates_screening_scope() -> None:
    with pytest.raises(ValueError, match="screening_scope must be"):
        SBMSPCov(screening_scope="sample")  # type: ignore[arg-type]


def test_chain_screening_scope_draws_and_publishes_one_fnr_support_per_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoffs = iter([jnp.asarray(0.0), jnp.asarray(2.0)])
    calls: list[jax.Array] = []

    def fake_cutoff(key: jax.Array, **_kwargs: object) -> jax.Array:
        calls.append(key)
        return next(cutoffs)

    monkeypatch.setattr(
        estimator_module,
        "pairwise_jeffreys_bayes_factors",
        lambda x: jnp.ones((x.shape[1], x.shape[1]), dtype=x.dtype),
    )
    monkeypatch.setattr(estimator_module, "estimate_fnr_cutoff", fake_cutoff)
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        n_chains=2,
        screening_scope="chain",
        n_cutoff_simulations=2,
        dtype="float32",
    )

    model.fit(_case(), key=jax.random.key(509))

    assert len(calls) == 2
    assert model.screening_cutoff_.shape == (2,)
    assert jnp.array_equal(model.screening_cutoff_, jnp.asarray([0.0, 2.0]))
    assert model.screening_mask_.shape == (2, 3, 3)
    assert bool(jnp.all(model.screening_mask_[0][~jnp.eye(3, dtype=jnp.bool_)]))
    assert not bool(jnp.any(model.screening_mask_[1]))
    assert model.initial_covariance_.shape == (2, 3, 3)
    assert model.posterior_samples_packed_.shape[:2] == (2, 1)
    assert model.diagnostics_.n_active_edges == (3, 0)
    assert model.diagnostics_.n_screened_edges == (0, 3)
