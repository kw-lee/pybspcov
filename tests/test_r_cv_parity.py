from pathlib import Path

import jax
import numpy as np

from pybspcov import cross_validate_band_ppp, cross_validate_threshold_ppp

FIXTURES = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _matrix(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",", skiprows=1, ndmin=2)


def _score_lookup(scores: np.ndarray) -> dict[tuple[float, float], float]:
    return {(row[0], row[1]): row[2] for row in scores}


def test_band_cv_scores_agree_with_public_r_1_0_3_run() -> None:
    expected = _matrix("band_cv_scores.csv")
    actual = cross_validate_band_ppp(
        _matrix("band_cv_x.csv"),
        bandwidths=[1, 2],
        epsilons=[0.01, 0.05],
        key=jax.random.key(701),
        n_samples=2000,
        dtype="float64",
        device="cpu",
    )

    expected_lookup = _score_lookup(expected)
    actual_lookup = _score_lookup(np.asarray(actual.scores))
    assert actual.best_bandwidth == int(expected[0, 0])
    for parameters, score in expected_lookup.items():
        np.testing.assert_allclose(actual_lookup[parameters], score, atol=0.2)


def test_threshold_cv_scores_agree_with_public_r_1_0_3_run() -> None:
    expected = _matrix("threshold_cv_scores.csv")
    actual = cross_validate_threshold_ppp(
        _matrix("threshold_cv_x.csv"),
        thresholds=[0.05, 0.2],
        epsilons=[0.01, 0.05],
        key=jax.random.key(709),
        n_samples=2000,
        n_folds=10,
        dtype="float64",
        device="cpu",
    )

    expected_lookup = _score_lookup(expected)
    actual_lookup = _score_lookup(np.asarray(actual.scores))
    assert actual.best_threshold == expected[0, 0]
    for parameters, score in expected_lookup.items():
        np.testing.assert_allclose(actual_lookup[parameters], score, atol=0.3)
