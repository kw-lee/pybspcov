from types import SimpleNamespace

import jax
import jax.numpy as jnp

from pybspcov import model_selection


def test_band_cross_validation_matches_r_1_0_3_fold_mean(monkeypatch) -> None:
    fold_scores = iter([jnp.asarray(-3.0), jnp.asarray(-6.0), jnp.asarray(-9.0)])

    class StubBandPPP:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fit(self, training, *, key):
            del training, key
            return SimpleNamespace(
                posterior_samples_=jnp.broadcast_to(jnp.eye(2), (1, 1, 2, 2))
            )

    monkeypatch.setattr(model_selection, "BandPPP", StubBandPPP)
    monkeypatch.setattr(
        model_selection,
        "_log_predictive_density",
        lambda _draws, _observation: next(fold_scores),
    )

    result = model_selection.cross_validate_band_ppp(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        bandwidths=[1],
        epsilons=[0.0],
        n_samples=1,
        key=jax.random.key(1),
        dtype="float32",
    )

    assert result.scores[0, 2] == -6.0
